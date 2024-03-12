import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]


def gradient_cv():
    img = cv.imread("../Picture/lingxiaohua.jpg", cv.IMREAD_GRAYSCALE)
    dx, dy = cv.spatialGradient(img)
    magnitude, angle = cv.cartToPolar(np.float32(dx), np.float32(dy))
    sobel_edgebw = magnitude > 0.20 * np.max(magnitude)
    img_smooth = cv.GaussianBlur(img, ksize=(0, 0), sigmaX=2, sigmaY=2)
    scharr_x = cv.Scharr(img_smooth, ddepth=cv.CV_64F, dx=1, dy=0)
    scharr_y = cv.Scharr(img_smooth, ddepth=cv.CV_64F, dx=0, dy=1)
    scharr_edge = cv.sqrt(scharr_x ** 2 + scharr_y ** 2)
    scharr_edgebw = scharr_edge > 0.20 * np.max(scharr_edge)

    plt.figure(figsize=(12, 5))
    plt.gray()
    plt.suptitle("使用OpenCV进行图像梯度计算", fontdict={"size": 15})

    plt.subplot(1, 3, 1)
    plt.imshow(img, vmin=0, vmax=255)
    plt.title("原图像", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 3, 2)
    plt.imshow(dx, vmin=0, vmax=255)
    plt.title("梯度分量dx绝对值", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 3, 3)
    plt.imshow(dy, vmin=0, vmax=255)
    plt.title("梯度分量dy绝对值", fontdict={'size': 15})
    plt.axis(False)

    plt.show()

    plt.figure(figsize=(12, 5))
    plt.gray()
    plt.suptitle("使用OpenCV进行图像梯度计算", fontdict={'size': 15})

    plt.subplot(1, 3, 1)
    plt.imshow(angle, vmin=0, vmax=255)
    plt.title("梯度的方向角", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 3, 2)
    plt.imshow(magnitude, vmin=0, vmax=255)
    plt.title("梯度幅值", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 3, 3)
    plt.imshow(sobel_edgebw, vmin=0, vmax=255)
    plt.title("梯度幅值的阈值分割", fontdict={'size': 15})
    plt.axis(False)

    plt.show()

    plt.figure(figsize=(12, 5))
    plt.gray()
    plt.suptitle("使用OpenCV计算图像梯度", fontdict={'size': 15})

    plt.subplot(1, 3, 1)
    plt.imshow(img_smooth, vmin=0, vmax=255)
    plt.title("高斯平滑滤波", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 3, 2)
    plt.imshow(scharr_edge, vmin=0, vmax=255)
    plt.title("Scharr")


if __name__ == "__main__":
    gradient_cv()
