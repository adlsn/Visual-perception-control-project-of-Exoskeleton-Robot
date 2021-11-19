import pyrealsense2 as rs
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calc_dist(xyz):
    x, y, z = xyz
    return (x**2 + y**2 + z**2)**0.5

pipe = rs.pipeline()
pf = pipe.start()

dev = pf.get_device()

dimgs = []
plt.ion()
fig = plt.figure()
while True:
    frame = pipe.wait_for_frames()
    color_rs = frame.get_color_frame()
    img = np.asanyarray(color_rs.get_data())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dimg = frame.get_depth_frame()


    '''获取距离'''
    dframe = dimg.as_depth_frame()
    # print(dframe.get_distance(5, 5))

    '''对齐'''
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frame)
    color_frame = aligned_frames.get_color_frame()
    aligned_depth_frame = aligned_frames.get_depth_frame()

    depth_image=np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    color_image=cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
    # color_image=color_image.astype(np.uint16)
    depth_image=depth_image.astype(np.uint8)
    # outimage=np.concatenate((color_image,depth_image),axis=0)
    # outimage=np.hstack((color_image,depth_image))

    mix=cv2.addWeighted(color_image,0.5,depth_image,0.5,0)
    cv2.imshow('1',mix)
    # cv2.imshow('2',depth_image)
    cv2.waitKey(1)

    '''点云获取'''
    # pc = rs.pointcloud()
    # pc.map_to(color_rs)
    # points_rs = pc.calculate(dimg)
    # points = np.asanyarray(points_rs.get_vertices())
    #
    # dists = list(map(calc_dist, points))
    # dists = np.array(dists).reshape(np.asanyarray(dimg.get_data()).shape)
    #
    # plt.cla()
    # plt.clf()
    #
    #
    # ax = fig.add_subplot(111)
    # ax.set_title('colorMap')
    # plt.imshow(dists)
    # ax.set_aspect('equal')
    #
    # cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    # cax.get_xaxis().set_visible(False)
    # cax.get_yaxis().set_visible(False)
    # cax.patch.set_alpha(0)
    # cax.set_frame_on(False)
    # plt.colorbar(orientation='vertical')
    # plt.show()
    #
    #
    # plt.pause(0.1)


    '''深度图对象转灰度图'''
    # depth = np.asanyarray(dimg.get_data())
    # dimg_gray = cv2.convertScaleAbs(depth, alpha=255 / 4000)
    # dimg = cv2.applyColorMap(dimg_gray, cv2.COLORMAP_JET)

    '''滤波'''
    # dimg = cv2.medianBlur(dimg, 13)
    # dimg = cv2.morphologyEx(dimg, cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8), iterations=10)
    # cv2.imshow('1', dimg)
    # key = cv2.waitKey(1)

    '''获取内参'''
    # profile = pf.get_stream(rs.stream.depth)
    # intr = profile.as_video_stream_profile().get_intrinsics()
    # print(intr)
    # if key == 27:
    #     cv2.destroyAllWindows()
    #     break
