import torchvision

from D_CNN import GlobalAvgPool2d, NetUnit1, NetUnit2, ConvTUnit, vgg_block
import torch
from torch import nn, optim
import numpy as np
import torch.nn.functional as F
from torchviz import make_dot
import d2lzh_pytorch as d2l
import time, logging, os
from tqdm import tqdm
import torch.utils.data as Data
from PIL import Image

logging.basicConfig(filename='log_D.txt', level=logging.DEBUG, format='%(filename)s - %(message)s')


def evaluate_accuracy(data_iter, net, device=None):
    """该函数用于双流语义分割神经网络，输入的神经网络有两个输入参数"""
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, x1, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device),x1.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
            else:  # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


class DRDNet(nn.Module):
    """
    场景识别语义分割模型，由重新设计的NIN残差神经网络和转置卷积神经网络构成
    该神经网络模型主要用于处理512*512及以上大小的图像
    labels是单通道图像，通道轴不用加入数据维度中
    """

    def __init__(self, conv_arch, **kwargs):
        super().__init__(**kwargs)
        self.layer1 = nn.Sequential()
        self.layer2 = nn.Sequential()
        self.layer3 = nn.Sequential()
        self.layer4 = nn.Sequential()
        self.layer5 = nn.Sequential()

        self.layer1.add_module('L1_IVNet_1', NetUnit1(3, 6, 11, 1, 5))
        self.layer1.add_module('L1_IVNet_2', NetUnit1(6, 6, 5, 1, 2))
        self.layer1.add_module('L1_VNet_5', NetUnit2(6, 6, 11, 2, 5))
        self.layer1.add_module('L1_BNLayer', nn.BatchNorm2d(6))

        self.layer2.add_module('L2_IVNet_1', NetUnit1(6, 6, 5, 1, 2))
        self.layer2.add_module('L2_IVNet_2', NetUnit1(6, 16, 5, 1, 2))
        self.layer2.add_module('L2_VNet_6', NetUnit2(16, 16, 5, 2, 2))
        self.layer2.add_module('L2_VNet_7', NetUnit2(16, 16, 5, 2, 2))
        self.layer2.add_module('L2_BNLayer', nn.BatchNorm2d(16))

        self.layer3.add_module('L3_IVNet_1', NetUnit1(16, 16, 3, 1, 1))
        self.layer3.add_module('L3_IVNet_2', NetUnit1(16, 32, 3, 1, 1))
        self.layer3.add_module('L3_VNet_8', NetUnit2(32, 32, 3, 2, 1))
        self.layer3.add_module('L3_VNet_9', NetUnit2(32, 64, 3, 2, 1))
        self.layer3.add_module('L3_VNet_10', NetUnit2(64, 128, 3, 2, 1))
        self.layer3.add_module('L3_VNet_11', NetUnit2(128, 256, 3, 2, 1))
        self.layer3.add_module('L3_VNet_12', NetUnit2(256, 512, 3, 2, 1))
        self.layer3.add_module('L3_BNLayer', nn.BatchNorm2d(512))

        self.layer4.add_module('L4_CT_1', ConvTUnit(512, 256, (3, 3), 1, 1))
        self.layer4.add_module('L4_CT_2', ConvTUnit(256, 128, (3, 3), 1, 1))
        self.layer4.add_module('L4_CT_3', ConvTUnit(128, 128, (3, 3), 1, 1))
        self.layer4.add_module('L4_CT_4', ConvTUnit(128, 64, (3, 3), 1, 1))
        self.layer4.add_module('L4_CT_5', ConvTUnit(64, 32, (3, 3), 1, 1))
        self.layer4.add_module('L4_CT_6', ConvTUnit(32, 16, (3, 3), 1, 1))
        self.layer4.add_module('L4_CT_7', ConvTUnit(16, 8, (3, 3), 1, 1))
        self.layer4.add_module('L4_CT_8', ConvTUnit(8, 4, (3, 3), 1, 1))

        for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
            self.layer5.add_module(f'vgg_block_{i + 1}', vgg_block(num_convs, in_channels,
                                                                   out_channels))
        self.layer5.add_module('L5_BNLayer', nn.BatchNorm2d(512))

    def forward(self, x, x1):
        self.out1 = self.layer1(x)
        self.out2 = self.layer2(self.out1)
        self.out3 = self.layer3(self.out2)
        self.out4 = self.out3 + self.layer5(x1)
        self.out5 = self.layer4(self.out4)
        return self.out5


def DNetTrain(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    """
    该函数专门用来训练DNet
    损失函数使用优化过的CROSSEntropyLoss，可直接将四维数据以及三维标签输入
    """
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("training on ", device)

    net = net.to(device)

    with tqdm(total=40, bar_format='{postfix[0]}: {postfix[5][epoch]} | {postfix[1]}: {postfix[5][loss]} | '
                                   '{postfix[2]}: {postfix[5][train_acc]} | {postfix[3]}: {postfix[5][test_acc]} | '
                                   '{postfix[4]}: {postfix[5][time]}',
              postfix=['epoch', 'loss', 'train_acc', 'test_acc', 'time',
                       dict(epoch=0, loss=0, train_acc=0, test_acc=0,
                            time=0)]) as t:

        batch_count = 0
        for epoch in range(num_epochs):
            train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
            for X, x1, y in train_iter:
                X = X.to(device)
                y = y.to(device)
                x1 = x1.to(device)
                y_hat = net(X, x1)
                l = loss(y_hat, y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                train_l_sum += l.cpu().item()
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item() / (512 ** 2)
                n += y.shape[0]
                batch_count += 1
            test_acc = evaluate_accuracy(test_iter, net) / (512 ** 2)

            t.postfix[5]['epoch'] = epoch + 1
            t.postfix[5]['loss'] = round(train_l_sum / batch_count, 4)
            t.postfix[5]['train_acc'] = round(train_acc_sum / n, 4)
            t.postfix[5]['test_acc'] = round(test_acc, 4)
            t.postfix[5]['time'] = round(time.time() - start, 2)
            t.update()

            logging.debug(f'epoch: {epoch + 1}, loss: {round(train_l_sum / batch_count, 4)}, '
                          f'train_acc: {train_acc_sum / n}, test_acc: {test_acc}, '
                          f'time: {round(time.time() - start, 2)}')

            # print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
            #       % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


# __________该部分代码实现标签到类别的映射___________

# COLORMAP = []
# CLASSES = ['flatland', 'grass', 'upstairs', 'downstairs']
# Colormap2Label = torch.zeros(512 ** 3, dtype=torch.uint8)
# for i, colormap in enumerate(COLORMAP):
#     Colormap2Label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
#
#
def label_indices(colormap, colormap2label):
    colormap = np.array(colormap.convert("RGB")).astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]


# __________数据预处理___________

def rand_crop(feature, label, height, width):
    feature = feature.permute(2, 0, 1)
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(
        feature, output_size=(height, width))

    feature = torchvision.transforms.functional.crop(feature, i, j, h, w)
    label = torchvision.transforms.functional.crop(label, i, j, h, w)

    feature = feature.permute(1, 2, 0)
    return feature, label


# ___________数据读取____________
class SceDataset(Data.Dataset):
    def __init__(self, is_train, path1, path2, crop_size, voc_dir, colormap2label, max_num=None):
        self.rgb_mean = np.array([0.485, 0.456, 0.406])

        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.tsf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.rgb_mean,
                                             std=self.rgb_std)
        ])

        self.crop_size = crop_size  # (h, w)

        features = []
        labels = []
        feature_list = os.listdir(path1)
        label_list = os.listdir(path2)

        for feature in feature_list:
            features.append(torch.from_numpy(np.array(Image.open(path1 + '\\' + feature).convert('RGB'))))

        for label in label_list:
            labels.append(torch.from_numpy(np.array(Image.open(path2 + '\\' + label).convert('RGB'))))

        self.features = self.filter(features)  # PIL image
        self.labels = self.filter(labels)  # PIL image
        self.colormap2label = colormap2label
        print('read ' + str(len(self.features)) + ' valid examples')

    def filter(self, imgs):
        return [img for img in imgs if (
                img.shape[0] >= self.crop_size[0] and
                img.shape[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = rand_crop(self.features[idx], self.labels[idx],
                                   *self.crop_size)

        return (self.tsf(feature),  # float32 tensor
                label_indices(label, self.colormap2label))  # uint8 tensor

    def __len__(self):
        return len(self.features)


if __name__ == '__main__':
    # ________测试_________
    conv_arch = ((1, 1, 8), (1, 8, 16), (1, 16, 32), (1, 32, 64), (1, 64, 128),
                 (2, 128, 128), (2, 128, 256), (2, 256, 512))
    # start = time.perf_counter()
    # net = DRDNet(conv_arch)
    # net.eval()
    # x1 = torch.randn(1, 1, 512, 512)
    # x = torch.randn(1, 3, 512, 512)
    # y = net(x, x1)
    # print(y.shape)
    # print(time.perf_counter() - start, 's')
    # g = make_dot(y)
    # g.render('DRDNet', view=True)

    # _________训练函数测试_________
    loss = torch.nn.CrossEntropyLoss()
    net = DRDNet(conv_arch)
    net = nn.DataParallel(net)
    features = torch.randn(100, 3, 512, 512)
    features1 = torch.randn(100, 1, 512, 512)
    labels = torch.randint(0, 4, (100, 512, 512))
    data = Data.TensorDataset(features, features1, labels)
    data_iter = Data.DataLoader(data, 32, shuffle=True)

    train_iter = data_iter
    test_iter = data_iter
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 100
    DNetTrain(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

    # ________数据预处理测试________
    # image = torch.randn(512, 512, 3)
    # label = torch.randint(0, 5, (512, 512))
    # f, l = rand_crop(image, label, 100, 100)
    # print(f.shape, l.shape)
