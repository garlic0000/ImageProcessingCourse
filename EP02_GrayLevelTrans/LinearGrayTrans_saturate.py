import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from skimage import io, exposure


def plot_grayHist(img, rows, cols, idx):
    # 绘制直方图
    plt.subplot(rows, cols, idx)
    histogram, bins, patch = plt.hist(img.ravel(), 256, histtype='bar', density=True)
    plt.xlabel('gray level')
    plt.ylabel('pixel percentage')
    plt.axis([0, 255, 0, np.max(histogram)])


def drawlinegraph(img, row, cols, idx):
    # 绘制折线图
    plt.subplot(row, cols, idx)
    histogram, bins, patch = plt.hist(img.ravel(), 256, histtype='step', density=True)
    plt.xlabel('gray level')
    plt.ylabel('pixel percentage')
    plt.axis([0, 255, 0, np.max(histogram)])


def plot_showpicture(img, name, rows, cols, idx):
    # 显示图片
    plt.subplot(rows, cols, idx)
    plt.imshow(img, vmin=0, vmax=255)  # 灰度图像
    plt.title(name)
    plt.axis(False)


def lineargraytrans_skimage():
    img = io.imread("../Picture/lingxiaohua.jpg", cv.IMREAD_GRAYSCALE)
    img_rescale1 = exposure.rescale_intensity(img)
    rlow_p2, rhigh_p98 = np.percentile(img, (2, 98))
    img_rescale2 = exposure.rescale_intensity(img, in_range=(rlow_p2, rhigh_p98))
    plt.figure(figsize=(15, 8))
    plt.gray()
    plot_showpicture(img, "img", 2, 3, 1)
    plot_showpicture(img_rescale1, "img_rescale1", 2, 3, 2)
    plot_showpicture(img_rescale2, "img_rescale2", 2, 3, 3)
    plot_grayHist(img, 2, 3, 4)
    plot_grayHist(img_rescale1, 2, 3, 5)
    plot_grayHist(img_rescale2, 2, 3, 6)
    plt.show()


def showhist_step():
    # 显示直方图 折线
    img = cv.imread("../Picture/lingxiaohua.jpg", cv.IMREAD_GRAYSCALE)
    img1 = cv.imread("../Picture/dark_lingxiaohua.jpg", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("../Picture/bright_lingxiaohua.jpg", cv.IMREAD_GRAYSCALE)
    plt.figure(figsize=(18, 8))
    plt.gray()
    plot_showpicture(img, "img", 2, 3, 1)
    plot_showpicture(img1, "img1", 2, 3, 2)
    plot_showpicture(img2, "img2", 2, 3, 3)
    drawlinegraph(img, 2, 3, 4)
    drawlinegraph(img1, 2, 3, 5)
    drawlinegraph(img2, 2, 3, 6)
    plt.show()


if __name__ == "__main__":
    lineargraytrans_skimage()
    showhist_step()

