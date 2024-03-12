import cv2 as cv
from skimage import util
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology, filters
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]


def smooth_cv():
    img = cv.imread("../Picture/cloud1.jpg", cv.IMREAD_GRAYSCALE)
    img_noise = util.random_noise(img, mode='gaussian', var=0.01)
    img_noise = util.img_as_ubyte(img_noise)
    # 双边滤波
    img_result1 = cv.bilateralFilter(img_noise, d=9, sigmaColor=50, sigmaSpace=100)
    # 3×3均值滤波
    img_result2 = cv.blur(img_noise, ksize=(3, 3))
    # 15×15均值滤波
    img_result3 = cv.blur(img_noise, ksize=(15, 15))
    # 3×3高斯滤波
    img_result4 = cv.GaussianBlur(img_noise, (3, 3), sigmaX=0, sigmaY=0)
    # 15×15高斯滤波
    img_result5 = cv.GaussianBlur(img_noise, (15, 15), sigmaX=0, sigmaY=0)

    plt.figure(figsize=(25, 12))
    plt.gray()
    plt.suptitle("使用OpenCV进行图像平滑", fontdict={'size': 15})

    plt.subplot(2, 4, 1)
    plt.imshow(img, vmin=0, vmax=255)
    plt.title("原图", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 4, 2)
    plt.imshow(img_noise, vmin=0, vmax=255)
    plt.title("添加高斯噪声的图像", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 4, 3)
    plt.imshow(img_result1, vmin=0, vmax=255)
    plt.title("双边滤波", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 4, 4)
    plt.imshow(img_result2, vmin=0, vmax=255)
    plt.title("3×3均值滤波", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 4, 5)
    plt.imshow(img_result3, vmin=0, vmax=255)
    plt.title("15×15均值滤波", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 4, 6)
    plt.imshow(img_result4, vmin=0, vmax=255)
    plt.title("3×3高斯滤波", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 4, 7)
    plt.imshow(img_result5, vmin=0, vmax=255)
    plt.title("15×15高斯滤波", fontdict={'size': 15})
    plt.axis(False)

    plt.show()


def roi_sk():
    img = cv.imread("../Picture/cat1.jpg", cv.IMREAD_GRAYSCALE)
    # 创建掩膜图像
    img_mask = np.zeros(img.shape[0:2], np.uint8)
    cv.rectangle(img_mask, (70, 75), (150, 150), 255, -1)

    img_result = img.copy()
    selem = morphology.square(25)
    img_roi = filters.rank.mean(img, selem, mask=img_mask)
    img_result[img_mask > 0] = img_roi[img_mask > 0]

    plt.figure(figsize=(18, 12))
    plt.gray()
    plt.suptitle("对图像中的兴趣区域进行平滑模糊滤波", fontdict={'size': 15})

    plt.subplot(2, 2, 1)
    plt.imshow(img, vmin=0, vmax=255)
    plt.title("原图", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 2, 2)
    plt.imshow(img_mask, vmin=0, vmax=255)
    plt.title("兴趣区域的掩膜图像", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 2, 3)
    plt.imshow(img_roi, vmin=0, vmax=255)
    plt.title("兴趣区域平滑", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 2, 4)
    plt.imshow(img_result, vmin=0, vmax=255)
    plt.title("脸部区域模糊", fontdict={'size': 15})
    plt.axis(False)

    plt.show()


if __name__ == "__main__":
    smooth_cv()
    roi_sk()
