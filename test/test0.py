import numpy as np
import cv2 as cv
from scipy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift
from skimage import io, util
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]


def pltshow(img1, img2, img3):
    plt.figure(figsize=(12, 10))
    plt.suptitle("采用巴特沃斯低通滤波器进行图像平滑", fontdict={'size': 15})

    plt.subplot(2, 2, 1)
    plt.imshow(img1)
    plt.title("原图", fontdict={'size': 15})
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(img2)
    plt.title("n=2  D0=30", fontdict={'size': 15})
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(img3)
    plt.title("n=2  D0=100", fontdict={'size': 15})
    plt.axis("off")

    plt.show()


# img=cv.imread('E:/tuxiangchuli/colorfulcat.jpg',0)
img = cv.imread('../Picture/colorfulcat.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
M, N = img.shape[0:2]
imgex = np.pad(img, ((0, M), (0, N), (0, 0)), mode='reflect')
# Fp=fftshift(fft2(imgex))
Fp = fft2(imgex, axes=(0, 1))
Fp = fftshift(Fp, axes=(0, 1))

DO = 30
v = np.arange(-N, N)
u = np.arange(-M, M)
Va, Ua = np.meshgrid(v, u)
Da2 = Ua ** 2 + Va ** 2
HGlpf = np.exp(-Da2 / (2 * DO ** 2))
HGlpf_3D = np.dstack((HGlpf, HGlpf, HGlpf))
HGhpf1 = 1 - np.exp(-Da2 / (2 * DO ** 2))
HGhpf_3D1 = np.dstack((HGhpf1, HGhpf1, HGhpf1))

Gp = Fp * HGlpf_3D
Gp1 = Fp * HGhpf_3D1
Gp = ifftshift(Gp, axes=(0, 1))
Gp1 = ifftshift(Gp1, axes=(0, 1))
imgp = np.real(ifft2(Gp, axes=(0, 1)))
imgp1 = np.real(ifft2(Gp1, axes=(0, 1)))
imgout = imgp[0:M, 0:N]
imgout1 = imgp1[0:M, 0:N]
imgout = np.uint8(np.clip(imgout, 0, 255))
imgout1 = np.uint8(np.clip(imgout1, 0, 255))
cv.imshow('yuantu', img)
cv.imshow('ditongresult', imgout)
cv.imshow('gaotongresult1', imgout1)
cv.waitKey(0)
cv.destroyAllWindows()
pltshow(img, imgout, imgout1)
