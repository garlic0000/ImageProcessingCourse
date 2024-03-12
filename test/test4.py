import cv2 as cv
import numpy as np
from scipy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from skimage import util
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]


# 1.读取一幅图像，对其进行傅立叶变换，并显示其原始频谱图和中心化后的频谱图。
def flytrans_cv():
    img = cv.imread("../Picture/tomato1.jpg", cv.IMREAD_GRAYSCALE)
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

    plt.figure(figsize=(18, 6))
    plt.gray()
    plt.suptitle("图像的傅里叶变换", fontdict={'size': 15})

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title("原图", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 3, 2)
    plt.imshow(imgdft_mag_log, cmap='gray', vmin=0, vmax=255)
    plt.title("原始频谱图", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 3, 3)
    plt.imshow(imgdft_mag_cent_log, cmap='gray', vmin=0, vmax=255)
    plt.title("中心化后的频谱图", fontdict={'size': 15})
    plt.axis(False)

    plt.show()


# 2.读取一幅图像，并添加高斯白噪声到其中，然后根据频域滤波流程并采用低通巴特沃斯滤波器对其进行滤波。

def butterworthtrans_low(img, n, D0):
    M, N = img.shape[0:2]
    imgex = np.pad(img, ((0, M), (0, N), (0, 0)), mode='reflect')
    Fp = fft2(imgex, axes=(0, 1))
    Fp = fftshift(Fp, axes=(0, 1))

    v = np.arange(-N, N)
    u = np.arange(-M, M)
    Va, Ua = np.meshgrid(v, u)
    Da2 = Ua ** 2 + Va ** 2
    HBlpf = 1 / (1 + (Da2 / D0 ** 2) ** n)
    HBlpf_3D = np.dstack((HBlpf, HBlpf, HBlpf))

    Gp = Fp * HBlpf_3D
    Gp = ifftshift(Gp, axes=(0, 1))
    imgp = np.real(ifft2(Gp, axes=(0, 1)))
    imgp = np.uint8(np.clip(imgp, 0, 255))
    imgout = imgp[0:M, 0:N]

    return imgout


def butterworth_readdata_low():
    img = cv.imread("../Picture/potato1.jpg", cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    n = 2
    D0 = 30
    imgout1 = butterworthtrans_low(img, n, D0)
    n = 2
    D0 = 100
    imgout2 = butterworthtrans_low(img, n, D0)
    n = 20
    D0 = 30
    imgout3 = butterworthtrans_low(img, n, D0)

    plt.figure(figsize=(12, 10))
    plt.suptitle("采用巴特沃斯低通滤波器进行图像平滑", fontdict={'size': 15})

    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title("原图", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 2, 2)
    plt.imshow(imgout1)
    plt.title("n=2  D0=30", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 2, 3)
    plt.imshow(imgout2)
    plt.title("n=2  D0=100", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 2, 4)
    plt.imshow(imgout3)
    plt.title("n=20  D0=30", fontdict={'size': 15})
    plt.axis(False)

    plt.show()


def butterworthtrans_high(img, n, D0):
    M, N = img.shape
    imgex = np.pad(img, ((0, M), (0, N)), mode='reflect')
    Fp = fftshift(fft2(imgex))
    v = np.arange(-N, N)
    u = np.arange(-M, M)
    Va, Ua = np.meshgrid(v, u)
    Da2 = Ua ** 2 + Va ** 2
    HBlpf = 1 - 1 / (1 + (Da2 / D0 ** 2) ** n)

    Gp = Fp * HBlpf
    Gp = ifftshift(Gp)
    imgp = np.real(ifft2(Gp))
    imgout = imgp[0:M, 0:N]
    imgout = np.uint8(np.clip(imgout, 0, 255))
    return imgout


def butterworth_readdata_high():
    img = cv.imread("../Picture/potato1.jpg", cv.IMREAD_GRAYSCALE)
    n = 2
    D0 = 30
    imgout1 = butterworthtrans_high(img, n, D0)
    n = 2
    D0 = 100
    imgout2 = butterworthtrans_high(img, n, D0)
    n = 20
    D0 = 30
    imgout3 = butterworthtrans_high(img, n, D0)

    plt.figure(figsize=(12, 10))
    plt.gray()
    plt.suptitle("采用巴特沃斯高通滤波器进行图像平滑", fontdict={'size': 15})

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title("原图", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 2, 2)
    plt.imshow(imgout1, cmap='gray', vmin=0, vmax=255)
    plt.title("n=2  D0=30", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 2, 3)
    plt.imshow(imgout2, cmap='gray', vmin=0, vmax=255)
    plt.title("n=2  D0=100", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 2, 4)
    plt.imshow(imgout3, cmap='gray', vmin=0, vmax=255)
    plt.title("n=20  D0=30", fontdict={'size': 15})
    plt.axis(False)

    plt.show()


def gaostrans_low(img, D0):
    M, N = img.shape
    imgex = np.pad(img, ((0, M), (0, N)), mode='reflect')
    Fp = fftshift(fft2(imgex))
    v = np.arange(-N, N)
    u = np.arange(-M, M)
    Va, Ua = np.meshgrid(v, u)
    Da2 = Ua ** 2 + Va ** 2
    HBlpf = np.exp(-Da2 / (2 * D0 ** 2))

    Gp = Fp * HBlpf
    Gp = ifftshift(Gp)
    imgp = np.real(ifft2(Gp))
    imgout = imgp[0:M, 0:N]
    imgout = np.uint8(np.clip(imgout, 0, 255))
    return imgout


def gaosi_readdata_low():
    img = cv.imread("../Picture/tomato1.jpg", cv.IMREAD_GRAYSCALE)
    D0 = 30
    imgout1 = gaostrans_low(img, D0)
    D0 = 50
    imgout2 = gaostrans_low(img, D0)
    D0 = 100
    imgout3 = gaostrans_low(img, D0)

    plt.figure(figsize=(12, 10))
    plt.gray()
    plt.suptitle("采用高斯低通滤波器进行图像平滑", fontdict={'size': 15})

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title("原图", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 2, 2)
    plt.imshow(imgout1, cmap='gray', vmin=0, vmax=255)
    plt.title("D0=30", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 2, 3)
    plt.imshow(imgout2, cmap='gray', vmin=0, vmax=255)
    plt.title("D0=50", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 2, 4)
    plt.imshow(imgout3, cmap='gray', vmin=0, vmax=255)
    plt.title("D0=100", fontdict={'size': 15})
    plt.axis(False)

    plt.show()


def gaostrans_high(img, D0):
    M, N = img.shape
    imgex = np.pad(img, ((0, M), (0, N)), mode='reflect')
    Fp = fftshift(fft2(imgex))
    v = np.arange(-N, N)
    u = np.arange(-M, M)
    Va, Ua = np.meshgrid(v, u)
    Da2 = Ua ** 2 + Va ** 2
    HBlpf = 1 - np.exp(-Da2 / (2 * D0 ** 2))

    Gp = Fp * HBlpf
    Gp = ifftshift(Gp)
    imgp = np.real(ifft2(Gp))
    imgout = imgp[0:M, 0:N]
    imgout = np.uint8(np.clip(imgout, 0, 255))
    return imgout


def gaosi_readdata_high():
    img = cv.imread("../Picture/tomato1.jpg", cv.IMREAD_GRAYSCALE)
    D0 = 30
    imgout1 = gaostrans_high(img, D0)
    D0 = 50
    imgout2 = gaostrans_high(img, D0)
    D0 = 100
    imgout3 = gaostrans_high(img, D0)

    plt.figure(figsize=(12, 10))
    plt.gray()
    plt.suptitle("采用高斯高通滤波器进行图像平滑", fontdict={'size': 15})

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title("原图", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 2, 2)
    plt.imshow(imgout1, cmap='gray', vmin=0, vmax=255)
    plt.title("D0=30", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 2, 3)
    plt.imshow(imgout2, cmap='gray', vmin=0, vmax=255)
    plt.title("D0=50", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 2, 4)
    plt.imshow(imgout3, cmap='gray', vmin=0, vmax=255)
    plt.title("D0=100", fontdict={'size': 15})
    plt.axis(False)

    plt.show()


def laplasi(img, alpha):
    img = util.img_as_float(img)
    M, N = img.shape
    imgex = np.pad(img, ((0, M), (0, N)), mode='reflect')
    Fp = fftshift(fft2(imgex))
    v = np.arange(-N, N)
    u = np.arange(-M, M)
    Va, Ua = np.meshgrid(v, u)
    Da2 = Ua ** 2 + Va ** 2
    HLap = -4 * np.pi * np.pi * Da2

    Gp = Fp * HLap
    Gp = ifftshift(Gp)
    imgp = np.real(ifft2(Gp))
    imgLap = imgp[0:M, 0:N]
    imgLap = imgLap / (np.max(np.abs(imgLap)))
    imgout = img - alpha * imgLap
    imgout = util.img_as_ubyte(np.clip(imgout, 0, 1))
    return imgout, imgLap


def laplasireaddata():
    img = cv.imread("../Picture/tomato1.jpg", cv.IMREAD_GRAYSCALE)
    alpha = 5
    imgout, imglap = laplasi(img, alpha)

    plt.figure(figsize=(18, 6))
    plt.gray()
    plt.suptitle("采用拉普拉斯算子进行图像锐化", fontdict={'size': 15})

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title("原图", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 3, 2)
    plt.imshow(imgout, cmap='gray', vmin=0, vmax=255)
    plt.title("拉普拉斯算子锐化", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 3, 3)
    plt.imshow(imglap, cmap='gray', vmin=0, vmax=255)
    plt.title("拉普拉斯边缘图像", fontdict={'size': 15})
    plt.axis(False)

    plt.show()


if __name__ == "__main__":
    # flytrans_cv()
    butterworth_readdata_low()
    # gaosi_readdata_low()
    # butterworth_readdata_high()
    # gaosi_readdata_high()
    # laplasireaddata()
