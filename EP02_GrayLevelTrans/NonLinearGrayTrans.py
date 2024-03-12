import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]


def gammacorrection(img, gamma):
    lookUpTable = np.zeros(256, np.uint8)
    for i in range(256):
        s = np.power(i / 255.0, gamma) * 255
        lookUpTable[i] = np.clip(s, 0, 255)
    img_out = lookUpTable[img]
    return img_out


def logarithmiccorrection(img):
    imgdft = np.fft.fft2(img)
    imgdft_cent = np.fft.fftshift(imgdft)
    imgdftmag_cent = np.abs(imgdft_cent)
    c = 255.0 / np.log10(1 + np.max(imgdftmag_cent))
    imgdftmaglog_cent = c * np.log10(imgdftmag_cent + 1)
    imgdftmaglog_cent = np.clip(imgdftmaglog_cent, 0, 255).astype(np.uint8)
    return imgdftmag_cent, imgdftmaglog_cent


def exponentialcorrection(img, k):
    c = 255.0 / (np.power(k, np.max(img) - 1))
    lookUpTable = np.zeros((1, 256), np.uint8)
    for i in range(256):
        s = c * np.abs(np.power(k, i) - 1)
        lookUpTable[0, i] = np.clip(s, 0, 255)
    img_out = cv.LUT(img, lookUpTable)
    return img_out


def showimage_gamma():
    img1 = cv.imread("../Picture/bright_cat1.jpg", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("../Picture/dark_cat2.jpg", cv.IMREAD_GRAYSCALE)
    img_out1 = gammacorrection(img1, gamma=10 / 3)
    img_out2 = gammacorrection(img2, gamma=1 / 3)
    plt.figure(figsize=(20, 12))
    plt.gray()

    plt.subplot(2, 2, 1)
    plt.imshow(img1, vmin=0, vmax=255)
    plt.title("猫", fontdict={'size': 25})
    plt.axis(False)

    plt.subplot(2, 2, 2)
    plt.imshow(img_out1, vmin=0, vmax=255)
    plt.title("伽马校正后的猫", fontdict={'size': 25})
    plt.axis(False)

    plt.subplot(2, 2, 3)
    plt.imshow(img2, vmin=0, vmax=255)
    plt.title("猫", fontdict={'size': 25})
    plt.axis(False)

    plt.subplot(2, 2, 4)
    plt.imshow(img_out2, vmin=0, vmax=255)
    plt.title("伽马校正的猫", fontdict={'size': 25})
    plt.axis(False)

    plt.show()


def showimage_log():
    img1 = cv.imread("../Picture/cat1.jpg", cv.IMREAD_GRAYSCALE)
    img_f, img_out1 = logarithmiccorrection(img1)

    plt.figure(figsize=(18, 6))
    plt.gray()

    plt.subplot(1, 3, 1)
    plt.imshow(img1, vmin=0, vmax=255)
    plt.title("猫", fontdict={'size': 25})
    plt.axis(False)

    plt.subplot(1, 3, 2)
    plt.imshow(img_f, vmin=0, vmax=255)
    plt.title("图像的傅里叶频谱", fontdict={'size': 25})
    plt.axis(False)

    plt.subplot(1, 3, 3)
    plt.imshow(img_out1, vmin=0, vmax=255)
    plt.title("处理后的傅里叶频谱", fontdict={'size': 25})
    plt.axis(False)

    plt.show()


def showimage_exp():
    img1 = cv.imread("../Picture/bright_cat1.jpg", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("../Picture/dark_cat2.jpg", cv.IMREAD_GRAYSCALE)
    k1 = 1.01
    k2 = 1/np.sqrt(1.01)
    img_out1 = exponentialcorrection(img1, k1)
    img_out2 = exponentialcorrection(img2, k2)
    plt.figure(figsize=(20, 12))
    plt.gray()

    plt.subplot(2, 2, 1)
    plt.imshow(img1, vmin=0, vmax=255)
    plt.title("猫", fontdict={'size': 25})
    plt.axis(False)

    plt.subplot(2, 2, 2)
    plt.imshow(img_out1, vmin=0, vmax=255)
    plt.title("指数校正后的猫", fontdict={'size': 25})
    plt.axis(False)

    plt.subplot(2, 2, 3)
    plt.imshow(img2, vmin=0, vmax=255)
    plt.title("猫", fontdict={'size': 25})
    plt.axis(False)

    plt.subplot(2, 2, 4)
    plt.imshow(img_out2, vmin=0, vmax=255)
    plt.title("经过指数校正的猫", fontdict={'size': 25})
    plt.axis(False)

    plt.show()


if __name__ == "__main__":
    showimage_gamma()
    showimage_log()
    showimage_exp()
