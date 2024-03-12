import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]


def spatialfilter_cv():
    # 有问题
    img_gray = cv.imread("../Picture/cloud1.jpg", cv.IMREAD_GRAYSCALE)
    img = cv.imread("../Picture/cloud1.jpg", cv.IMREAD_COLOR)
    img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    kav5 = np.ones((5, 5), np.float32) / 25
    # 边界镜像
    img_gray_smoothed1 = cv.filter2D(img_gray, -1, kernel=kav5, borderType=cv.BORDER_REFLECT)
    # 填充常值
    img_gray_smoothed2 = cv.filter2D(img_gray, -1, kernel=kav5, borderType=cv.BORDER_CONSTANT)

    img_RGB_smoothed1 = cv.filter2D(img_RGB, -1, kernel=kav5, borderType=cv.BORDER_REFLECT)
    img_RGB_smoothed2 = cv.filter2D(img_RGB, -1, kernel=kav5, borderType=cv.BORDER_CONSTANT)

    plt.figure(figsize=(18, 12))
    plt.suptitle("opencv线性空域滤波器", fontdict={'size': 15})

    plt.subplot(2, 3, 1)
    plt.gray()
    plt.imshow(img_gray, vmin=0, vmax=255)
    plt.title("原图", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 2)
    plt.gray()
    plt.imshow(img_gray_smoothed1, vmin=0, vmax=255)
    plt.title("边界镜像", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 3)
    plt.gray()
    plt.imshow(img_gray_smoothed2, vmin=0, vmax=255)
    plt.title("填充常值0", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 4)
    plt.imshow(img_RGB)
    plt.title("原图", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 5)
    plt.imshow(img_RGB_smoothed1)
    plt.title("边界镜像", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 6)
    plt.imshow(img_RGB_smoothed2)
    plt.title("填充常值0", fontdict={'size': 15})
    plt.axis(False)

    plt.show()


def spatialfilter_sci():
    img_gray = cv.imread("../Picture/cloud1.jpg", cv.IMREAD_GRAYSCALE)
    img = cv.imread("../Picture/cloud1.jpg", cv.IMREAD_COLOR)
    img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    kav5_gray = np.ones((5, 5), np.float32) / 25
    kav5_RGB = np.ones((5, 5, 5), np.float32) / 125

    # 边界反射
    img_gray_smooth1 = ndimage.convolve(img_gray, kav5_gray, mode='reflect')
    # 填充常值 0
    img_gray_smooth2 = ndimage.convolve(img_gray, kav5_gray, mode='constant', cval=0)

    img_RGB_smooth1 = ndimage.convolve(img_RGB, kav5_RGB, mode='reflect')
    img_RGB_smooth2 = ndimage.convolve(img_RGB, kav5_RGB, mode='constant', cval=0)

    plt.figure(figsize=(18, 12))
    plt.suptitle("SciPy线性空域滤波器", fontdict={'size': 15})

    plt.subplot(2, 3, 1)
    plt.gray()
    plt.imshow(img_gray, vmin=0, vmax=255)
    plt.title("原图", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 2)
    plt.gray()
    plt.imshow(img_gray_smooth1, vmin=0, vmax=255)
    plt.title("边界反射", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 3)
    plt.gray()
    plt.imshow(img_gray_smooth2, vmin=0, vmax=255)
    plt.title("填充常值0", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 4)
    plt.imshow(img_RGB)
    plt.title("原图", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 5)
    plt.imshow(img_RGB_smooth1)
    plt.title("边界反射", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 6)
    plt.imshow(img_RGB_smooth2)
    plt.title("填充常值0", fontdict={'size': 15})
    plt.axis(False)

    plt.show()


if __name__ == "__main__":
    spatialfilter_cv()
    spatialfilter_sci()
