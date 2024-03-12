import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, util, color
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]


def imagesharp_cv():
    img = cv.imread("../Picture/lingxiaohua.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    # 构造4-邻域 8-邻域拉普拉斯算子
    klap4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    klap8 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    # 定义强度因子
    alpha = 1.5
    img_smooth = cv.GaussianBlur(img, ksize=(5, 5), sigmaX=0, sigmaY=0)
    img_klap4 = cv.filter2D(img_smooth, -1, kernel=klap4)
    img_sharpen4 = cv.addWeighted(img, 1, img_klap4, -1 * alpha, 0, dtype=-1)
    img_klap8 = cv.filter2D(img_smooth, -1, kernel=klap8)
    img_sharpen8 = cv.addWeighted(img, 1, img_klap8, -1 * alpha, 0, dtype=-1)

    plt.figure(figsize=(30, 6))
    plt.gray()
    plt.suptitle("用拉普拉斯算子进行图像锐化", fontdict={'size': 15})

    plt.subplot(1, 5, 1)
    plt.imshow(img, vmin=0, vmax=255)
    plt.title("原图像", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 5, 2)
    plt.imshow(img_klap4, vmin=0, vmax=255)
    plt.title("4-邻域拉普拉斯边缘", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 5, 3)
    plt.imshow(img_sharpen4, vmin=0, vmax=255)
    plt.title("4-邻域拉普拉斯算子锐化结果", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 5, 4)
    plt.imshow(img_klap8, vmin=0, vmax=255)
    plt.title("8-邻域拉普拉斯边缘", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 5, 5)
    plt.imshow(img_sharpen8, vmin=0, vmax=255)
    plt.title("8-邻域拉普拉斯算子锐化结果", fontdict={'size': 15})
    plt.axis(False)

    plt.show()


def unsharpmask_sharp_cv():
    img = cv.imread("../Picture/cloud1.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    sigma = 1.5
    alpha = 2
    T = 0.01
    img_smooth = cv.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    img_edge = cv.subtract(img, img_smooth)
    img_edge[np.abs(img_edge) < T * 255] = 0
    img_usm = cv.addWeighted(img, 1, img_edge, alpha, gamma=0, dtype=-1)

    plt.figure(figsize=(18, 6))
    plt.gray()
    plt.suptitle("使用OpenCV实现钝化掩膜图像锐化", fontdict={'size': 15})

    plt.subplot(1, 3, 1)
    plt.imshow(img, vmin=0, vmax=255)
    plt.title("原图像", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 3, 2)
    plt.imshow(img_edge, vmin=0, vmax=255)
    plt.title("边缘图像", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 3, 3)
    plt.imshow(img_usm, vmin=0, vmax=255)
    plt.title("钝化掩膜图像锐化结果", fontdict={'size': 15})
    plt.axis(False)

    plt.show()


def unsharkmask_sharp_np():
    img = cv.imread("../Picture/cat1.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    sigma = 1.5
    alpha = 2
    T = 0.01
    img_smooth = cv.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    img_edge = img - img_smooth
    img_edge[np.abs(img_edge) < T * 255] = 0
    img_usm = img + alpha * img_edge
    img_usm = np.clip(img_usm, 0, 255).astype(np.uint8)

    plt.figure(figsize=(18, 6))
    plt.gray()
    plt.suptitle("使用numpy实现钝化掩膜图像锐化", fontdict={'size': 15})

    plt.subplot(1, 3, 1)
    plt.imshow(img, vmin=0, vmax=255)
    plt.title("原图像", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 3, 2)
    plt.imshow(img_edge, vmin=0, vmax=255)
    plt.title("边缘图像", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 3, 3)
    plt.imshow(img_usm, vmin=0, vmax=255)
    plt.title("钝化掩膜图像锐化结果", fontdict={'size': 15})
    plt.axis(False)

    plt.show()


def unsharpmask_sk():
    img1 = cv.imread("../Picture/lingxiaohua.jpg", cv.IMREAD_GRAYSCALE)
    img2_bgr = cv.imread("../Picture/lingxiaohua.jpg", cv.IMREAD_COLOR)
    img2_rgb = cv.cvtColor(img2_bgr, cv.COLOR_BGR2RGB)
    img1_usm = filters.unsharp_mask(img1, radius=2.0, amount=2.0)
    img1_usm = util.img_as_ubyte(img1_usm)
    img2_hsv = color.rgb2hsv(img2_rgb)
    img2_hsv[:, :, 2] = filters.unsharp_mask(img2_hsv[:, :, 2], radius=1.5, amount=2.0)
    img2_usm = color.hsv2rgb(img2_hsv)
    img2_usm = util.img_as_ubyte(img2_usm)

    plt.figure(figsize=(24, 6))
    plt.suptitle("使用skimage库函数进行钝化掩膜图像锐化", fontdict={'size': 15})

    plt.subplot(1, 4, 1)
    plt.gray()
    plt.imshow(img1, vmin=0, vmax=255)
    plt.title("灰度图像", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 4, 2)
    plt.gray()
    plt.imshow(img1_usm, vmin=0, vmax=255)
    plt.title("USM锐化结果", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 4, 3)
    plt.imshow(img2_rgb)
    plt.title("彩色图像", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 4, 4)
    plt.imshow(img2_usm)
    plt.title("USM锐化结果", fontdict={'size': 15})
    plt.axis(False)

    plt.show()


def sharp_halo_sk():
    img = cv.imread("../Picture/tomato1.jpg", cv.IMREAD_GRAYSCALE)
    img1_usm = filters.unsharp_mask(img, radius=0.5, amount=10)
    img2_usm = filters.unsharp_mask(img, radius=20, amount=20)
    img3_usm = filters.unsharp_mask(img, radius=10, amount=0.8)
    img1_usm = util.img_as_ubyte(img1_usm)
    img2_usm = util.img_as_ubyte(img2_usm)
    img3_usm = util.img_as_ubyte(img3_usm)

    plt.figure(figsize=(24, 6))
    plt.gray()
    plt.suptitle("钝化掩膜图像锐化时的光晕现象", fontdict={'size': 15})

    plt.subplot(1, 4, 1)
    plt.imshow(img, vmin=0, vmax=255)
    plt.title("原图像", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 4, 2)
    plt.imshow(img1_usm, vmin=0, vmax=255)
    plt.title("标准差=0.5 强度因子=10", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 4, 3)
    plt.imshow(img2_usm, vmin=0, vmax=255)
    plt.title("标准差=10 强度因子=10", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 4, 4)
    plt.imshow(img3_usm, vmin=0, vmax=255)
    plt.title("标准差=10 强度因子=0.8", fontdict={'size': 15})
    plt.axis(False)

    plt.show()


# 水平微分
# 垂直微分

if __name__ == "__main__":
    imagesharp_cv()
    unsharpmask_sharp_cv()
    unsharkmask_sharp_np()
    unsharpmask_sk()
    sharp_halo_sk()
