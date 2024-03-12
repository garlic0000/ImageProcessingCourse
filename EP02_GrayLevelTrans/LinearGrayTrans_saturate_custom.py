import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def transf(r, rlow, rhigh, slow, shigh):
    # 带饱和处理的线性灰度变换函数
    if r < rlow:
        return rlow
    elif r >= rlow and r < rhigh:
        return (shigh - slow) / (rhigh - rlow) * (r - rlow) + slow
    else:
        return shigh


def chooselowhigh(img, lowpercent, highpercent):
    # 选择rlow rhigh 使用百分比的方式来进行选择
    # 小于等于rlow 的像素占lowpercent
    # 大于等于rhigh的像素占highpercent
    # 对于8位256级灰度图像 通常令slow=0 shigh=255
    # 将二维的图像像素 数组 转成 一维 对其进行排序 asc 找到前 n%
    onedim = []  # 存放一维数据
    for m in range(len(img)):  # 二维转一维
        for n in range(len(img[0])):
            onedim.append(img[m][n])
    onedim = sorted(onedim)  # 升序排列 sorted不会改变源列表 需要使用返回值
    lowpos = int(lowpercent * len(onedim))  # 根据比例确定下标
    highpos = int((1 - highpercent) * len(onedim))
    return onedim[lowpos], onedim[highpos]


def grayleveltrans(img, rlow, rhigh, slow, shigh):
    # 将图像对应的位置的像素进行转换
    img1 = []  # 转换后的图像  存放转换后的像素值
    for m in range(len(img)):
        imgl = []  # 存放每一行的像素
        for n in range(len(img[0])):
            pixel = transf(img[m][n], rlow, rhigh, slow, shigh)
            imgl.append(pixel)
        img1.append(imgl)
    return img1


def main():
    # 读取灰度图像 有时会有三个通道 在读的时候使用灰度方式读取
    img = cv.imread("../Picture/gray_lingxiaohua.png", cv.IMREAD_GRAYSCALE)
    # 不能输入2%
    rlow, rhigh = chooselowhigh(img, 0.02, 0.02)
    img1 = grayleveltrans(img, rlow, rhigh, 0, 255)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("img")
    plt.axis(False)

    plt.subplot(1, 2, 2)
    plt.imshow(img1)
    plt.title("img1")
    plt.axis(False)

    plt.show()


if __name__ == "__main__":
    main()
