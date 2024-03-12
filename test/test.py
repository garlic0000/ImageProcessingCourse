import cv2 as cv
import numpy as np
from scipy import ndimage
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]


def cal_fourier_frequency_sci():
    img = cv.imread("../Picture/tomato1.jpg", cv.IMREAD_GRAYSCALE)
    imgdft = fft2(img)
    imgdft_mag = np.abs(imgdft)
    imgdft_cent = fftshift(imgdft)  # 可能是浮点数 需要转换
    imgdft_mag_cent = np.abs(imgdft_cent)
    # 对数增强
    c1 = 255 / np.log10(1 + np.max(imgdft_mag))
    imgdft_mag_log = c1 * np.log10(1 + imgdft_mag)
    c2 = 255 / np.log10(1 + np.max(imgdft_mag_cent))
    imgdft_mag_cent_log = c2 * np.log10(1 + imgdft_mag_cent)

    plt.figure(figsize=(12, 10))
    plt.gray()
    plt.suptitle("采用SciPy中的fft2函数计算灰度图形的傅里叶频谱", fontdict={'size': 15})

    plt.subplot(2, 2, 1)
    plt.imshow(img, vmin=0, vmax=255)
    plt.title("原图像", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 2, 2)
    plt.imshow(imgdft_mag_cent, vmin=0, vmax=255)
    plt.title("中心化幅度谱", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 2, 3)
    plt.imshow(imgdft_mag_cent_log, vmin=0, vmax=255)
    plt.title("对数校正的中心化幅度谱", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 2, 4)
    plt.imshow(imgdft_mag_log, vmin=0, vmax=255)
    plt.title("对数校正的未中心化幅度谱", fontdict={'size': 15})
    plt.axis(False)

    plt.show()


if __name__ == "__main__":
    cal_fourier_frequency_sci()
