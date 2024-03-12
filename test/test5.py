import cv2 as cv
import numpy as np
from scipy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift
from skimage import util
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]


def AtmoTurlenceSim(img, k):
    rows, cols = img.shape[0:2]
    if img.ndim == 3:
        # 有问题
        imgx = np.pad(img, ((0, rows), (0, cols), (0, 0)), mode='reflect')
    elif img.ndim == 2:
        imgx = np.pad(img, ((0, rows), (0, cols)), mode='reflect')
    img_dft = fftshift(fft2(imgx, axes=(0, 1)), axes=(0, 1))
    v = np.arange(-cols, cols)
    u = np.arange(-rows, rows)
    Va, Ua = np.meshgrid(v, u)
    D2 = Ua ** 2 + Va ** 2
    Hatm = np.exp(-k * (D2 ** (5.0 / 6.0)))
    if img.ndim == 3:
        Hatm = np.dstack((Hatm, Hatm, Hatm))
    Gp = img_dft * Hatm
    Gp = ifftshift(Gp, axes=(0, 1))
    imgp = np.real(ifft2(Gp, axes=(0, 1)))
    imgp = np.uint8(np.clip(imgp, 0, 255))
    imgout = imgp[0:rows, 0:cols]
    return imgout, Hatm


def wiener_filter():
    img = cv.imread("../Picture/cat1.jpg", cv.IMREAD_GRAYSCALE)
    rows, cols = img.shape[0:2]
    img_deg, Hatm = AtmoTurlenceSim(img, 0.0025)
    img_deg_noi = util.random_noise(img_deg, mode='gaussian', var=0.001)
    img_deg_noi = util.img_as_ubyte(img_deg_noi)
    if img.ndim == 3:
        imgex = np.pad(img_deg_noi, ((0, rows), (0, cols), (0, 0)), mode='reflect')
    elif img.ndim == 2:
        imgex = np.pad(img_deg_noi, ((0, rows), (0, cols)), mode='reflect')
    img_dft = fftshift(fft2(imgex, axes=(0, 1)), axes=(0, 1))
    NSR = 0.005
    Gp = img_dft*np.conj(Hatm)/(np.abs(Hatm)**2+NSR+np.finfo(np.float32).eps)
    Gp = ifftshift(Gp, axes=(0, 1))
    imgp = np.real(ifft2(Gp, axes=(0, 1)))
    imgp = np.uint8(np.clip(imgp, 0, 255))
    img_res = imgp[0:rows, 0:cols]

    imgs = [img, img_deg, img_deg_noi, img_res]
    imgtitle = ["原图像", "大气湍流模糊退化 k=0.0025", "高斯噪声 var=0.001", "复原图像"]
    pos = 0
    fig, axe = plt.subplots(nrows=2, ncols=2)
    for i in range(2):
        for j in range(2):
            axe[i, j].imshow(imgs[pos], cmap='gray', vmin=0, vmax=255)
            axe[i, j].set_title(imgtitle[pos], fontsize=10)
            axe[i, j].axis(False)
            pos = pos + 1
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    wiener_filter()


