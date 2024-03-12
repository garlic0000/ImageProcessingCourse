import cv2 as cv
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]


def histmatch_gray():
    # 原图
    img = cv.imread("../Picture/cat1.jpg", cv.IMREAD_GRAYSCALE)
    # 参考图像
    img_m = cv.imread("../Picture/lingxiaohua.jpg", cv.IMREAD_GRAYSCALE)
    img_matched = exposure.match_histograms(img, img_m, channel_axis=None)
    img_matched = img_matched.astype(np.uint8)

    plt.figure(figsize=(18, 6))
    plt.gray()
    plt.suptitle("使用skimage进行灰度图像直方图匹配", fontdict={'size': 15})

    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title("原图", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 2)
    plt.imshow(img_m)
    plt.title("参考图片", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 3)
    plt.imshow(img_matched)
    plt.title("匹配后的图片", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 4)
    plt.hist(img.ravel(), bins=256, histtype='bar')
    plt.title("灰度直方图", fontdict={'size': 15})
    plt.xlabel("灰度值", fontdict={'size': 15})
    plt.ylabel("像素频数", fontdict={'size': 15})

    plt.subplot(2, 3, 5)
    plt.hist(img_m.ravel(), bins=256, histtype='bar')
    plt.title("灰度直方图", fontdict={'size': 15})
    plt.xlabel("灰度值", fontdict={'size': 15})
    plt.ylabel("像素频数", fontdict={'size': 15})

    plt.subplot(2, 3, 6)
    plt.hist(img_matched.ravel(), bins=256, histtype='bar')
    plt.title("灰度直方图", fontdict={'size': 15})
    plt.xlabel("灰度值", fontdict={'size': 15})
    plt.ylabel("像素频数", fontdict={'size': 15})

    plt.show()


def histmatch_color():
    img = cv.imread("../Picture/cat1.jpg", cv.IMREAD_COLOR)
    img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_m = cv.imread("../Picture/lingxiaohua.jpg", cv.IMREAD_COLOR)
    img_m_RGB = cv.cvtColor(img_m, cv.COLOR_BGR2RGB)

    img_matched = exposure.match_histograms(img_RGB, img_m_RGB, channel_axis=-1)

    plt.figure(figsize=(18, 6))
    plt.suptitle("使用skimage进行彩色图像直方图匹配", fontdict={'size': 15})

    plt.subplot(2, 3, 1)
    plt.imshow(img_RGB)
    plt.title("原图", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 2)
    plt.imshow(img_m_RGB)
    plt.title("参考图像", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 3)
    plt.imshow(img_matched)
    plt.title("匹配后的图像", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 4)
    plt.hist(img_RGB[:, :, 0].ravel(), bins=256, histtype='bar')
    plt.hist(img_RGB[:, :, 1].ravel(), bins=256, histtype='bar')
    plt.hist(img_RGB[:, :, 2].ravel(), bins=256, histtype='bar')
    plt.title("彩色图像直方图", fontdict={'size': 15})
    plt.xlabel("灰度值", fontdict={'size': 15})
    plt.ylabel("像素频数", fontdict={'size': 15})

    plt.subplot(2, 3, 5)
    plt.hist(img_m_RGB[:, :, 0].ravel(), bins=256, histtype='bar')
    plt.hist(img_m_RGB[:, :, 1].ravel(), bins=256, histtype='bar')
    plt.hist(img_m_RGB[:, :, 2].ravel(), bins=256, histtype='bar')
    plt.title("彩色图像直方图", fontdict={'size': 15})
    plt.xlabel("灰度值", fontdict={'size': 15})
    plt.ylabel("像素频数", fontdict={'size': 15})

    plt.subplot(2, 3, 6)
    plt.hist(img_matched[:, :, 0].ravel(), bins=256, histtype='bar')
    plt.hist(img_matched[:, :, 1].ravel(), bins=256, histtype='bar')
    plt.hist(img_matched[:, :, 2].ravel(), bins=256, histtype='bar')
    plt.title("彩色图像直方图", fontdict={'size': 15})
    plt.xlabel("灰度值", fontdict={'size': 15})
    plt.ylabel("像素频数", fontdict={'size': 15})

    plt.show()


if __name__ == "__main__":
    histmatch_gray()
    histmatch_color()
