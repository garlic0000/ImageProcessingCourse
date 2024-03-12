import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def read_showbase_RGB():
    # 读一幅彩色图像 查看图像的基本信息
    img = cv.imread("../Picture/cat1.jpg", cv.IMREAD_COLOR)  # BGR
    print("数组维数:", img.ndim)  # 输出图像数组的维数
    print("图像的高、宽及颜色分量数:", img.shape)
    print("图像数据类型:", img.dtype)
    return img  # 返回图像的像素矩阵 用于后续操作


def read_pixel_RGB(img):
    # 读取像素值
    pixb = img[100, 300, 0]  # 读取坐标(100, 300)处像素的B分量值 BGR
    print("坐标[100, 300]处像素的B分量值为:", pixb)
    pixbgr = img[100, 300, :]  # 读取坐标(100, 300)处像素的BGR值
    print("坐标[100, 300]处像素的BGR颜色分量值:", pixbgr)


def change_pixel_RGB(img):
    # 改写像素值
    img2 = img.copy()  # 不修改传进来的图像的像素值
    img2[100, 300, 1] = 200  # 将图像坐标为[100, 300]处的像素的G分量的值改为200
    img2[200:210, 200:300, :] = [0, 255, 0]  # 在图像中画一条200个像素长,10个像素宽的水平绿线
    return img2


def showRGBinplot(img):
    # 1 以灰度图像方式显示各个颜色的分量值 明暗程度对应各分量值的大小
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(16, 8))
    plt.gray()  # 以灰度图像显示二位数组

    plt.subplot(2, 3, 1)
    plt.imshow(imgRGB[:, :, 0], vmin=0, vmax=255)
    plt.title("R")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(imgRGB[:, :, 1], vmin=0, vmax=255)
    plt.title("G")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(imgRGB[:, :, 2], vmin=0, vmax=255)
    plt.title("B")
    plt.axis("off")

    # 2 以单色彩色图像方式显示RGB分量
    imgR = imgRGB.copy()
    imgR[:, :, 1:3] = 0  # [0, 1, 2]  1 2 置0
    imgG = imgRGB.copy()
    imgG[:, :, 0] = 0
    imgG[:, :, 2] = 0
    imgB = imgRGB.copy()
    imgB[:, :, 0:2] = 0

    plt.subplot(2, 3, 4)
    plt.imshow(imgR)
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(imgG)
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(imgB)
    plt.axis("off")

    plt.show()


def showpicincv2_RGB(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyWindow(name)


def showpicinplot_RGB(img1, img2):
    plt.figure(figsize=(12, 6))  # 创建一个宽12长6的窗口

    plt.subplot(1, 2, 1)  # 显示1行2列的子图 当前子图为第一个子图
    plt.imshow(img1[:, :, ::-1])  # 数组中所有颜色分量反过来  BGR -> RGB
    plt.title("img1")  # 子图1标题
    plt.axis("off")  # 不显示坐标网格

    plt.subplot(1, 2, 2)  # 显示1行2列的子图 当前子图为第2个子图
    plt.imshow(img2[:, :, ::-1])
    plt.title("img2")
    # plt.axis("off")

    plt.show()  # 显示多子图


def indexed_image_read_show():
    im = Image.open("../Picture/cat.gif")  # 读入一幅索引图像
    plt.figure(figsize=(12, 5))
    plt.imshow(im)
    plt.title("cat")
    plt.axis("off")
    plt.show()
    print("图片文件格式:", im.format)
    print("图像的高宽:", im.size)
    print("图像类型:", im.mode)
    return im  # 返回Image对象用于后续操作


def getIndexOfPic_indexedimage(im):
    imgX = np.array(im)
    print("颜色索引值数组的大小:", imgX.shape)
    print("颜色索引值数组的元素:", imgX)
    impalette = im.getpalette()
    num_colors = int(len(impalette)/3)
    cmaparray = np.array(impalette).reshape(num_colors, 3)
    print("图像调色板数组的大小:", cmaparray.shape)
    print("图像调色板数组的元素:", cmaparray)