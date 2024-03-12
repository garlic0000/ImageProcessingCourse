import cv2 as cv
import matplotlib.pyplot as plt
from skimage import exposure, util
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]


def histequal_cv():
    # 注意是灰度图像
    img = cv.imread("../Picture/cat1.jpg", cv.IMREAD_GRAYSCALE)
    img_equal1 = cv.equalizeHist(img)  # 全局
    retval = cv.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))  # 自适应
    img_equal2 = retval.apply(img)

    plt.figure(figsize=(24, 8))
    plt.gray()
    plt.suptitle("使用opencv进行直方图均衡化", fontdict={'size': 15})

    plt.subplot(2, 3, 1)
    plt.imshow(img, vmin=0, vmax=255)
    plt.title("原图", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 2)
    plt.imshow(img_equal1)
    plt.title("全局均衡化", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 3)
    plt.imshow(img_equal2)
    plt.title("自适应均衡化", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 4)
    plt.hist(img.ravel(), bins=256, histtype='bar')
    plt.title("原图 灰度直方图", fontdict={'size': 15})
    plt.xlabel("灰度值", fontdict={'size': 15})
    plt.ylabel("像素频数", fontdict={'size': 15})

    plt.subplot(2, 3, 5)
    plt.hist(img_equal1.ravel(), bins=256, histtype='bar')
    plt.title("全局均衡化 灰度直方图", fontdict={'size': 15})
    plt.xlabel("灰度值", fontdict={'size': 15})
    plt.ylabel("像素频数", fontdict={'size': 15})

    plt.subplot(2, 3, 6)
    plt.hist(img_equal2.ravel(), bins=256, histtype='bar')
    plt.title("自适应均衡化 灰度直方图", fontdict={'size': 15})
    plt.xlabel("灰度值", fontdict={'size': 15})
    plt.ylabel("像素频数", fontdict={'size': 15})

    plt.show()


def histequal_sk():
    img = cv.imread("../Picture/lingxiaohua.jpg", cv.IMREAD_GRAYSCALE)
    img_equal1 = exposure.equalize_hist(img)
    img_equal1 = util.img_as_ubyte(img_equal1)
    img_equal2 = exposure.equalize_adapthist(img, clip_limit=0.03)
    img_equal2 = util.img_as_ubyte(img_equal2)

    plt.figure(figsize=(18, 6))
    plt.gray()
    plt.suptitle("使用skimage进行直方图均衡化", fontdict={'size': 15})

    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title("原图", fontdict={"size": 15})
    plt.axis(False)

    plt.subplot(2, 3, 2)
    plt.imshow(img_equal1)
    plt.title("全局均衡化", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 3)
    plt.imshow(img_equal2)
    plt.title("自适应均衡化", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(2, 3, 4)
    plt.hist(img.ravel(), bins=256, histtype='bar')
    plt.title("灰度直方图", fontdict={'size': 15})
    plt.xlabel("灰度值", fontdict={'size': 15})
    plt.ylabel("像素频数", fontdict={'size': 15})

    plt.subplot(2, 3, 5)
    plt.hist(img_equal1.ravel(), bins=256, histtype='bar')
    plt.title("灰度直方图", fontdict={'size': 15})
    plt.xlabel("灰度值", fontdict={'size': 15})
    plt.ylabel("灰度频数", fontdict={'size': 15})

    plt.subplot(2, 3, 6)
    plt.hist(img_equal2.ravel(), bins=256, histtype='bar')
    plt.title("灰度直方图", fontdict={'size': 15})
    plt.xlabel("灰度值", fontdict={'size': 15})
    plt.ylabel("灰度频数", fontdict={'size': 15})

    plt.show()


if __name__ == "__main__":
    histequal_cv()
    histequal_sk()
