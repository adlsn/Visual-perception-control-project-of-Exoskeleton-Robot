import os, logging

import torchvision
from PIL import Image

from D_CNN import vgg_block
import torch
from torch import nn, optim
import numpy as np
import torch.nn.functional as F
from torchviz import make_dot
import d2lzh_pytorch as d2l
import time
from tqdm import tqdm
import torch.utils.data as Data

logging.basicConfig(filename='.\\log\\log.txt', level=logging.DEBUG, format='%(filename)s - %(message)s')


class VDNet(nn.Module):
    """
    该网络用于场景中目标障碍物分类检测，基础块使用VGG，结构为VGG-16；
    该网络主要用于检测室外环境的深度图像目标分类检测；
    类型分为有障碍物以及没有障碍物两类;
    障碍物目标检测：
    fc_features = 2048
    conv_arch = ((1, 1, 32), (1, 32, 64), (1, 64, 128), (2, 128, 128), (2, 128, 256), (2, 256, 512), (3, 512, 512))
    坡度分类检测：
    fc_features = 8192
    conv_arch = ((1, 1, 32), (1, 32, 64), (1, 64, 128), (2, 128, 128), (2, 128, 256), (2, 256, 512), (3, 512, 512))
    """

    def __init__(self, conv_arch, fc_features, fc_hidden_units=4096, **kwargs):
        super().__init__(**kwargs)

        self.conv = nn.Sequential()
        self.fc = nn.Sequential()
        for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
            self.conv.add_module(f'vgg_block_{i + 1}', vgg_block(num_convs, in_channels,
                                                                 out_channels))
        self.fc.add_module('fc', nn.Sequential(d2l.FlattenLayer(),
                                               nn.Linear(fc_features, int(fc_hidden_units / 2)),
                                               nn.BatchNorm1d(int(fc_hidden_units / 2)),
                                               nn.ReLU(),
                                               nn.Dropout(0.4),
                                               nn.Linear(int(fc_hidden_units / 2), int(fc_hidden_units / 2)),
                                               nn.BatchNorm1d(int(fc_hidden_units / 2)),
                                               nn.ReLU(),
                                               nn.Dropout(0.4),
                                               nn.Linear(int(fc_hidden_units / 2), int(fc_hidden_units / 2)),
                                               nn.BatchNorm1d(int(fc_hidden_units / 2)),
                                               nn.ReLU(),
                                               nn.Dropout(0.4),
                                               nn.Linear(int(fc_hidden_units / 2), fc_hidden_units),
                                               nn.BatchNorm1d(fc_hidden_units),
                                               nn.ReLU(),
                                               nn.Dropout(0.5),
                                               nn.Linear(fc_hidden_units, 2),
                                               nn.BatchNorm1d(2),
                                               nn.ReLU(),
                                               nn.Dropout(0.5)
                                               ))

    def forward(self, x):
        self.out1 = self.conv(x)
        self.out2 = self.fc(self.out1)
        return self.out2


class VDNetv2(nn.Module):
    """
    该网络用于场景中斜坡坡度检测，基础块使用VGG，结构为VGG-16加深度图双流；
    该网络主要用于检测室外环境的深度图像目标分类检测；
    类型分为0，3，6，9，12五种坡度;
    障碍物目标检测：
    fc_features = 2048
    conv_arch = ((1, 1, 32), (1, 32, 64), (1, 64, 128), (2, 128, 128), (2, 128, 256), (2, 256, 512), (3, 512, 512))
    坡度分类检测：
    fc_features = 8192
    conv_arch = ((1, 3, 32), (1, 32, 64), (1, 64, 128), (2, 128, 128), (2, 128, 256), (2, 256, 512), (3, 512, 512))
    dconv_arch = ((1, 1, 8), (1, 8, 16), (1, 16, 32), (1, 32, 64), (1, 64, 128), (1, 128, 256), (2, 256, 512))
    """

    def __init__(self, conv_arch, dconv_arch, fc_features, fc_hidden_units=4096, **kwargs):
        super().__init__(**kwargs)

        self.conv = nn.Sequential()
        self.fc = nn.Sequential()
        self.dconv = nn.Sequential()
        for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
            self.conv.add_module(f'vgg_block_{i + 1}', vgg_block(num_convs, in_channels,
                                                                 out_channels))

        for i, (num_convs, in_channels, out_channels) in enumerate(dconv_arch):
            self.dconv.add_module(f'vgg_block_{i + 1}', vgg_block(num_convs, in_channels,
                                                                  out_channels))

        self.fc.add_module('fc', nn.Sequential(d2l.FlattenLayer(),
                                               nn.Linear(fc_features, int(fc_hidden_units / 2)),
                                               nn.BatchNorm1d(int(fc_hidden_units / 2)),
                                               nn.ReLU(),
                                               nn.Dropout(0.1),
                                               nn.Linear(int(fc_hidden_units / 2), int(fc_hidden_units / 2)),
                                               nn.BatchNorm1d(int(fc_hidden_units / 2)),
                                               nn.ReLU(),
                                               nn.Dropout(0.1),
                                               nn.Linear(int(fc_hidden_units / 2), int(fc_hidden_units / 2)),
                                               nn.BatchNorm1d(int(fc_hidden_units / 2)),
                                               nn.ReLU(),
                                               nn.Dropout(0.1),
                                               nn.Linear(int(fc_hidden_units / 2), fc_hidden_units),
                                               nn.BatchNorm1d(fc_hidden_units),
                                               nn.ReLU(),
                                               nn.Dropout(0.001),
                                               nn.Linear(fc_hidden_units, 5),
                                               nn.BatchNorm1d(5),
                                               nn.ReLU(),
                                               nn.Dropout(0.001)
                                               ))

    def forward(self, x, x1):
        self.out1 = self.conv(x)
        self.out2 = self.dconv(x1)
        self.out3 = self.out1 + self.out2
        self.out = self.fc(self.out3)
        return self.out


def VGG_train(net, train_iter, test_iter, batch_size, optimizer, num_epochs):
    """专门用于训练VGG障碍物检测模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("training on ", device)
    with tqdm(total=40, bar_format="{postfix[0]}: {postfix[5][epoch]} | {postfix[1]}: {postfix[5][loss]} | "
                                   "{postfix[2]}: {postfix[5][train_acc]} | {postfix[3]}: {postfix[5][test_acc]} | "
                                   "{postfix[4]}: {postfix[5][time]}",
              postfix=['epoch', 'loss', 'train acc', 'test acc', 'time',
                       dict(epoch=0, loss=0, train_acc=0,
                            test_acc=0, time=0)]) as t:

        net = net.to(device)

        loss = torch.nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
            for X, y in train_iter:
                X = X.to(device)
                y = y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                train_l_sum += l.cpu().item()
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                n += y.shape[0]
                batch_count += 1
            test_acc = d2l.evaluate_accuracy(test_iter, net)

            t.postfix[5]['epoch'] = epoch + 1
            t.postfix[5]['loss'] = round(train_l_sum / batch_count, 4)
            t.postfix[5]['train_acc'] = train_acc_sum / n
            t.postfix[5]['test_acc'] = test_acc
            t.postfix[5]['time'] = round(time.time() - start, 2)
            t.update()
            # print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
            #       % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
    print('_____Training finished!_____')


class V_dataiter(Data.Dataset):

    def __init__(self, rgbDir, depthDir):
        self.depth_mean = 4206.46
        self.depth_std = 787.54
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])

        self.tsf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.rgb_mean,
                                             std=self.rgb_std)
        ])

        self.rgb = []
        self.depth = []
        self.label = []

        for label in tqdm(os.listdir(rgbDir)):

            for imgname in tqdm(os.listdir(rgbDir + '\\' + label)):
                self.rgb.append(np.array(Image.open(rgbDir + '\\' + label + '\\' + imgname)))
                self.depth.append(self.filter(depthDir + '\\' + label + '\\' + 'depth' + imgname[3:]))

                label1 = int(label)
                label1 = torch.tensor(label1)
                self.label.append(label1)

        print(f'load {len(self.rgb)} data.', '\n')

    def filter(self, dimg_dir):
        dimg = np.array(Image.open(dimg_dir).convert('L'))
        dimg = dimg[np.newaxis, :]
        dimg = dimg.astype(np.float32)
        dimg = torch.tensor(dimg, dtype=torch.float32)
        return (dimg - self.depth_mean) / self.depth_std

    def __getitem__(self, idx):
        x = self.tsf(self.rgb[idx])
        x1 = self.depth[idx]
        y = self.label[idx]
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(x, (256, 256))
        x = torchvision.transforms.functional.crop(x, i, j, h, w)
        x1 = torchvision.transforms.functional.crop(x1, i, j, h, w)

        return x, x1, y

    def __len__(self):
        return len(self.rgb)


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, X1, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                acc_sum += (net(X.cuda(), X1.cuda()).argmax(dim=1) == y.cuda()).float().sum().cpu().item()
                net.train()  # 改回训练模式
            else:  # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def VGG_trainv2(net, train_iter, test_iter, batch_size, optimizer, num_epochs):
    """专门用于训练VGGv2坡度检测模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("training on ", device)
    with tqdm(total=40, bar_format="{postfix[0]}: {postfix[5][epoch]} | {postfix[1]}: {postfix[5][loss]} | "
                                   "{postfix[2]}: {postfix[5][train_acc]} | {postfix[3]}: {postfix[5][test_acc]} | "
                                   "{postfix[4]}: {postfix[5][time]}",
              postfix=['epoch', 'loss', 'train acc', 'test acc', 'time',
                       dict(epoch=0, loss=0, train_acc=0,
                            test_acc=0, time=0)]) as t:

        net = net.to(device)

        loss = torch.nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
            for X, X1, y in train_iter:
                X = X.cuda()
                X1 = X1.cuda()
                y = y.cuda()
                y_hat = net(X, X1)
                l = loss(y_hat, y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                train_l_sum += l.cpu().item()
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                n += y.shape[0]
                batch_count += 1
            test_acc = evaluate_accuracy(test_iter, net)

            t.postfix[5]['epoch'] = epoch + 1
            t.postfix[5]['loss'] = round(train_l_sum / batch_count, 4)
            t.postfix[5]['train_acc'] = round(train_acc_sum / n, 4)
            t.postfix[5]['test_acc'] = round(test_acc, 3)
            t.postfix[5]['time'] = round(time.time() - start, 2)
            t.update()
            # print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
            #       % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

            if epoch % 500 == 0:
                torch.save(net.module.state_dict(), f'.\\params\\epoch_{epoch // 500}.pt')

            logging.debug(f'epoch: {epoch + 1}, loss: {round(train_l_sum / batch_count, 4)}, '
                          f'train_acc: {round(train_acc_sum / n, 2)}, test_acc: {round(test_acc, 2)}, '
                          f'time: {round(time.time() - start, 2)}')
    print('_____Training finished!_____')


if __name__ == '__main__':
    # _____________障碍物检测分类参数______________
    # fc_features = 2048
    # conv_arch = ((1, 1, 32), (1, 32, 64), (1, 64, 128), (2, 128, 128), (2, 128, 256), (2, 256, 512), (3, 512, 512))

    # start = time.perf_counter()
    # net = VDNet(conv_arch, fc_features).cuda()
    # net.eval()
    # x = torch.randn(5, 1, 256, 256).cuda()
    # y = net(x)
    # print(y.shape)
    # print(time.perf_counter() - start, 's')
    # g = make_dot(y)
    # g.render('VGG_Module', view=True)

    # ________________模型测试________________

    # net = VDNet(conv_arch, fc_features).cuda()
    #
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    # batch_size, num_epochs = 5, 100
    #
    # features = torch.randn(100, 1, 256, 256)
    # labels = torch.randint(0, 2, (100,))
    # data = Data.TensorDataset(features, labels)
    # train_iter = Data.DataLoader(data, 5, shuffle=True)
    # test_iter = Data.DataLoader(data, 5, shuffle=True)
    #
    # VGG_train(net, train_iter, test_iter, batch_size, optimizer, num_epochs)

    # _________________V2模型测试___________________

    fc_features = 2048
    conv_arch = ((1, 3, 32), (1, 32, 64), (1, 64, 128), (2, 128, 128), (2, 128, 256), (2, 256, 512), (3, 512, 512))
    dconv_arch = ((1, 1, 8), (1, 8, 16), (1, 16, 32), (1, 32, 64), (1, 64, 128), (1, 128, 256), (2, 256, 512))
    net = VDNetv2(conv_arch, dconv_arch, fc_features).cuda()
    net.load_state_dict(torch.load('epoch_1st.pt'))

    net = torch.nn.DataParallel(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.00000001, weight_decay=3)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.00001)
    batch_size, num_epochs = 32, 5000

    rgbDir = r'D:\adl_exp\train_data\rgb_data'
    depthDir = r'D:\adl_exp\train_data\depth_data'
    rgbDir1 = r'D:\adl_exp\test_data\rgb_data'
    depthDir1 = r'D:\adl_exp\test_data\depth_data'
    data_iter = V_dataiter(rgbDir, depthDir)
    testiter = V_dataiter(rgbDir1, depthDir1)

    train_iter = Data.DataLoader(data_iter, 32, shuffle=True)
    test_iter = Data.DataLoader(testiter, 12, shuffle=True)
    VGG_trainv2(net, train_iter, test_iter, batch_size, optimizer, num_epochs)
