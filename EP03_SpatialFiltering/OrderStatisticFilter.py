import cv2 as cv
from skimage import util, filters, morphology
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]


def max_min_filter_sci():
    img = cv.imread("../Picture/cloud1.jpg", cv.IMREAD_GRAYSCALE)
    # 添加椒噪声
    img_noise1 = util.random_noise(img, mode='pepper', amount=0.1)
    img_noise1 = util.img_as_ubyte(img_noise1)
    # 添加盐噪声
    img_noise2 = util.random_noise(img, mode='salt', amount=0.1)
    img_noise2 = util.img_as_ubyte(img_noise2)
    # 3维最大值滤波器
    img_result1 = ndimage.maximum_filter(img_noise1, size=3, mode='reflect')
    # 3维最小值滤波器
    img_result2 = ndimage.minimum_filter(img_noise2, size=3, mode='reflect')

    plt.figure(figsize=(18, 6))
    plt.gray()
    plt.suptitle("SciPy最大值最小值滤波器", fontdict={'size': 15})

    plt.subplot(2, 3, 1)
    plt.imshow(img, vmin=0, vmax=255)
    plt.title("原图", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 2)
    plt.imshow(img_noise1, vmin=0, vmax=255)
    plt.title("添加椒噪声", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 3)
    plt.imshow(img_noise2, vmin=0, vmax=255)
    plt.title("添加盐噪声", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 5)
    plt.imshow(img_result1, vmin=0, vmax=255)
    plt.title("3×3最大值滤波器", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 6)
    plt.imshow(img_result2, vmin=0, vmax=255)
    plt.title("3×3最小值滤波器", fontdict={'size': 15})
    plt.axis(False)

    plt.show()


def max_min_filter_sk():
    img = cv.imread("../Picture/lingxiaohua.jpg", cv.IMREAD_GRAYSCALE)
    img_noise1 = util.random_noise(img, mode='pepper', amount=0.1)
    img_noise1 = util.img_as_ubyte(img_noise1)
    img_noise2 = util.random_noise(img, mode='salt', amount=0.1)
    img_noise2 = util.img_as_ubyte(img_noise2)
    # selem = np.ones((3, 3), np.uint8)
    selem = morphology.square(3)
    img_result1 = filters.rank.maximum(img_noise1, selem)
    img_result2 = filters.rank.minimum(img_noise2, selem)

    plt.figure(figsize=(18, 12))
    plt.gray()
    plt.suptitle("sk最大值最小值滤波器", fontdict={'size': 15})

    plt.subplot(2, 3, 1)
    plt.imshow(img, vmin=0, vmax=255)
    plt.title("原图", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 2)
    plt.imshow(img_noise1, vmin=0, vmax=255)
    plt.title("添加椒噪声", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 3)
    plt.imshow(img_noise2, vmin=0, vmax=255)
    plt.title("添加盐噪声", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 5)
    plt.imshow(img_result1, vmin=0, vmax=255)
    plt.title("最大值滤波器", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 6)
    plt.imshow(img_result2, vmin=0, vmax=255)
    plt.title("最小值滤波器", fontdict={'size': 15})
    plt.axis(False)

    plt.show()


def median_filter_cv():
    img = cv.imread("../Picture/cloud1.jpg", cv.IMREAD_GRAYSCALE)
    img_noise = util.random_noise(img, mode='s&p', amount=0.2)
    img_noise = util.img_as_ubyte(img_noise)
    # 3×3中值滤波器
    img_result1 = cv.medianBlur(img_noise, ksize=3)
    # 5×5中值滤波器
    img_result2 = cv.medianBlur(img_noise, ksize=5)

    plt.figure(figsize=(12, 6))
    plt.gray()
    plt.suptitle("Opencv中值滤波", fontdict={'size': 15})

    plt.subplot(2, 2, 1)
    plt.imshow(img, vmin=0, vmax=255)
    plt.title("原图", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 2, 2)
    plt.imshow(img_noise, vmin=0, vmax=255)
    plt.title("添加椒盐噪声", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 2, 3)
    plt.imshow(img_result1, vmin=0, vmax=255)
    plt.title("3×3中值滤波", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 2, 4)
    plt.imshow(img_result2, vmin=0, vmax=255)
    plt.title("5×5中值滤波", fontdict={'size': 15})
    plt.axis(False)

    plt.show()


def max_min_median_filter_sci():
    img = cv.imread("../Picture/cat1.jpg", cv.IMREAD_GRAYSCALE)
    img_noise = util.random_noise(img, mode='s&p', amount=0.2)
    img_noise = util.img_as_ubyte(img_noise)
    # 3×3最大值滤波器
    img_result1 = ndimage.rank_filter(img_noise, rank=-1, size=3)
    # 3×3最小值滤波器
    img_result2 = ndimage.rank_filter(img_noise, rank=0, size=3)
    # 3×3中值滤波器
    img_result3 = ndimage.rank_filter(img_noise, rank=4, size=3)

    plt.figure(figsize=(18, 6))
    plt.gray()
    plt.suptitle("SciPy统计排序滤波器", fontdict={'size': 15})

    plt.subplot(2, 3, 1)
    plt.imshow(img, vmin=0, vmax=255)
    plt.title("原图", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 2)
    plt.imshow(img_noise, vmin=0, vmax=255)
    plt.title("添加椒盐噪声", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 4)
    plt.imshow(img_result1, vmin=0, vmax=255)
    plt.title("3×3最大值滤波", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 5)
    plt.imshow(img_result2, vmin=0, vmax=255)
    plt.title("3×3最小值滤波", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 6)
    plt.imshow(img_result3, vmin=0, vmax=255)
    plt.title("3×3中值滤波", fontdict={'size': 15})
    plt.axis(False)

    plt.show()


def adapt_filter_cv():
    img = cv.imread("../Picture/cloud1.jpg", cv.IMREAD_GRAYSCALE)
    img_noise = util.random_noise(img, mode="s&p", amount=0.2)
    img_noise = util.img_as_ubyte(img_noise)
    img_result1 = cv.medianBlur(img_noise, ksize=3)
    img_result2 = cv.medianBlur(img_noise, ksize=7)


if __name__ == "__main__":
    max_min_filter_sci()
    max_min_filter_sk()
    median_filter_cv()
    max_min_median_filter_sci()
