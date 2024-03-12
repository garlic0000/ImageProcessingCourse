import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fft2, fftshift
import cv2 as cv
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False


def frequency_analysis():
    Fs = 500
    T = 1 / Fs
    N = 1000
    t = np.linspace(0.0, N * T, N)

    S = 0.7 * np.sin(2 * np.pi * 50 * t) + 1 * np.sin(2 * np.pi * 120 * t)
    X = S + np.random.standard_normal(t.size)
    Yu = fft(X)
    Yu_amp2side = np.abs(Yu / N)
    Yu2 = fftshift(Yu)
    Yu_amp2side_centered = np.abs(Yu2 / N)
    Yu_amp1side = Yu_amp2side[0: int(N / 2)].copy()
    Yu_amp1side[1: -1] = 2 * Yu_amp1side[1: -1]

    plt.figure(figsize=(12, 10))
    plt.suptitle("一维含噪信号的频谱分析", fontdict={'size': 15})

    plt.subplot(2, 2, 1)
    plt.plot(1000 * t, X, color='black')
    plt.title('含噪信号的时域波形', fontdict={'size': 15})
    plt.xlabel("t(ms)", fontdict={'size': 15})
    plt.ylabel("X(t)", fontdict={'size': 15})

    f = np.linspace(0.0, Fs, N)
    plt.subplot(2, 2, 2)
    plt.plot(f, Yu_amp2side, color='black')
    plt.title("含噪信号的双边幅度谱(未中心化)", fontdict={'size': 15})
    plt.xlabel("f(Hz)", fontdict={'size': 15})
    plt.ylabel("|Yu_amp2side(f)|", fontdict={'size': 15})

    f = np.linspace(0.0, Fs, N) - Fs / 2
    plt.subplot(2, 2, 3)
    plt.plot(f, Yu_amp2side_centered, color='black')
    plt.title("含噪信号的中心化双边幅度谱", fontdict={'size': 15})
    plt.xlabel('f(Hz)', fontdict={'size': 15})
    plt.ylabel("|Yu_amp2side_centered|", fontdict={'size': 15})

    f = np.linspace(0.0, Fs / 2.0, N // 2)
    plt.subplot(2, 2, 4)
    plt.plot(f, Yu_amp1side, color='black')
    plt.title("含噪信号的单边幅度谱", fontdict={'size': 15})
    plt.xlabel("f(Hz)", fontdict={'size': 15})
    plt.ylabel("|Yu_amp1side(f)|", fontdict={'size': 15})

    plt.show()


def cal_fourier_frequency_sci():
    img = cv.imread("../Picture/tomato1.jpg", cv.IMREAD_GRAYSCALE)
    imgdft = fft2(img)
    imgdft_mag = np.abs(imgdft)
    imgdft_cent = fftshift(imgdft)
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


def cal_fourier_frequency_cv():
    img = cv.imread("../Picture/potato1.jpg", cv.IMREAD_GRAYSCALE)
    M, N = img.shape
    P = cv.getOptimalDFTSize(M)
    Q = cv.getOptimalDFTSize(N)
    imgn = cv.copyMakeBorder(img, top=0, bottom=P - M, left=0, right=Q - N, borderType=cv.BORDER_REPLICATE)
    imgdft = cv.dft(np.float32(imgn), flags=cv.DFT_COMPLEX_OUTPUT)
    imgdft_mag = cv.magnitude(imgdft[:, :, 0], imgdft[:, :, 1])
    imgdft_cent = fftshift(imgdft)
    imgdft_mag_cent = cv.magnitude(imgdft_cent[:, :, 0], imgdft_cent[:, :, 1])
    c1 = 255 / np.log10(1 + np.max(imgdft_mag))
    imgdft_mag_log = c1 * np.log10(1 + imgdft_mag)
    c2 = 255 / np.log10(1 + np.max(imgdft_mag_cent))
    imgdft_mag_cent_log = c2 * np.log10(1 + imgdft_mag_cent)

    plt.figure(figsize=(12, 10))
    plt.gray()
    plt.suptitle("采用OpenCV中dft函数计算灰度图像的傅里叶频谱", fontdict={'size': 15})

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
    frequency_analysis()
    cal_fourier_frequency_sci()
    cal_fourier_frequency_cv()
