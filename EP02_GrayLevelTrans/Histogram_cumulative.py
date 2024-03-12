import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]


def calhist_plt():
    img = cv.imread("../Picture/lingxiaohua.jpg", cv.IMREAD_COLOR)
    img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.title("使用matplotlib计算直方图", fontdict={'size': 20})

    plt.subplot(1, 2, 1)
    plt.imshow(img_RGB)
    plt.title("图片凌霄花", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 2, 2)
    plt.hist(img_RGB[:, :, 0].ravel(), bins=256, color='r')
    plt.hist(img_RGB[:, :, 1].ravel(), bins=256, color='g')
    plt.hist(img_RGB[:, :, 2].ravel(), bins=256, color='b')
    plt.title("颜色通道灰度图像直方图", fontdict={'size': 15})
    plt.xlabel("灰度值", fontdict={'size': 15})
    plt.ylabel("像素频数", fontdict={'size': 15})

    plt.show()


def calhist_cv():
    img = cv.imread("../Picture/lingxiaohua.jpg", cv.IMREAD_COLOR)
    img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    hist_r = cv.calcHist([img_RGB], [0], None, [256], [0, 256])
    hist_g = cv.calcHist([img_RGB], [1], None, [256], [0, 256])
    hist_b = cv.calcHist([img_RGB], [2], None, [256], [0, 256])

    plt.figure(figsize=(12, 6))
    plt.title("使用opencv计算的直方图", fontdict={'size': 20})

    plt.subplot(1, 2, 1)
    plt.imshow(img_RGB)
    plt.title("图片凌霄花", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 2, 2)
    plt.plot(hist_r, color='r')
    plt.plot(hist_g, color='g')
    plt.plot(hist_b, color='b')
    plt.title("彩色图像直方图", fontdict={'size': 15})
    plt.xlabel("灰度值", fontdict={'size': 15})
    plt.ylabel("像素频数", fontdict={'size': 15})

    plt.show()


def calhist_np():
    img = cv.imread("../Picture/lingxiaohua.jpg", cv.IMREAD_COLOR)
    img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    hist_r, bin_edges = np.histogram(img_RGB[:, :, 0], 256, range=(0.0, 255.0))
    hist_g, bin_edges = np.histogram(img_RGB[:, :, 1], 256, range=(0.0, 255.0))
    hist_b, bin_edges = np.histogram(img_RGB[:, :, 2], 256, range=(0.0, 255.0))

    cdf_r = np.cumsum(hist_r / img.size)
    cdf_g = np.cumsum(hist_g / img.size)
    cdf_b = np.cumsum(hist_b / img.size)

    plt.figure(figsize=(18, 6))
    plt.title("使用numpy计算的直方图和累积直方图", fontdict={'size': 20})

    plt.subplot(1, 3, 1)
    plt.imshow(img_RGB)
    plt.title("图片凌霄花", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 3, 2)
    plt.plot(hist_r, color='r')
    plt.plot(hist_g, color='g')
    plt.plot(hist_b, color='b')
    plt.title("彩色图像直方图", fontdict={'size': 15})
    plt.xlabel("灰度值", fontdict={'size': 15})
    plt.ylabel("像素频数", fontdict={'size': 15})

    plt.subplot(1, 3, 3)
    plt.plot(cdf_r, color='r')
    plt.plot(cdf_g, color='g')
    plt.plot(cdf_b, color='b')
    plt.title("累计直方图", fontdict={'size': 15})
    plt.xlabel("灰度值", fontdict={'size': 15})
    plt.ylabel("像素累计相对频数", fontdict={'size': 15})

    plt.show()


def calhist_sk():
    img = cv.imread("../Picture/lingxiaohua.jpg", cv.IMREAD_COLOR)
    img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    hist_r, bin_centers = exposure.histogram(img_RGB[:, :, 0], nbins=256, source_range='dtype', normalize=False)
    hist_g, bin_centers = exposure.histogram(img_RGB[:, :, 1], nbins=256, source_range='dtype', normalize=False)
    hist_b, bin_centers = exposure.histogram(img_RGB[:, :, 2], nbins=256, source_range='dtype', normalize=False)

    cdf_r, bin_centers = exposure.cumulative_distribution(img_RGB[:, :, 0], nbins=256)
    cdf_g, bin_centers = exposure.cumulative_distribution(img_RGB[:, :, 1], nbins=256)
    cdf_b, bin_centers = exposure.cumulative_distribution(img_RGB[:, :, 2], nbins=256)

    plt.figure(figsize=(30, 12))
    plt.suptitle("使用skimage计算的直方图和累积直方图", fontdict={'size': 20})

    plt.subplot(1, 3, 1)
    plt.imshow(img_RGB)
    plt.title("图像凌霄花", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 3, 2)
    plt.plot(hist_r, color='r')
    plt.plot(hist_g, color='g')
    plt.plot(hist_b, color='b')
    plt.xlabel("灰度值", fontdict={'size': 15})
    plt.ylabel("像素频数", fontdict={'size': 15})
    plt.title('灰度值频数直方图', fontdict={'size': 15})

    plt.subplot(1, 3, 3)
    plt.plot(cdf_r, color='r')
    plt.plot(cdf_g, color='g')
    plt.plot(cdf_b, color='b')
    plt.xlabel("灰度值", fontdict={'size': 15})
    plt.ylabel("像素累积相对频数", fontdict={'size': 15})
    plt.title("灰度值相对频数累积直方图")

    plt.show()


if __name__ == "__main__":
    calhist_plt()
    calhist_cv()
    calhist_np()
    calhist_sk()
