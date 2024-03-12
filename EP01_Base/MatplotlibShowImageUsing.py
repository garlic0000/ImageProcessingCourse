import matplotlib.pyplot as plt
import cv2 as cv
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]


def pltshow1():
    # 使用matplotlib进行图像显示
    # 读灰色图像时 最好指定cv.IMREAD_GRAYSCALE 否则会有三个通道
    img_bgr = cv.imread("../Picture/lingxiaohua.jpg", cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)  # plt显示rgb图像
    # 由于是三维数组 因此要将前三个都表示出来 即[:,:,::-1] 而不是[:-1]
    img_rgb1 = img_bgr[:, :, ::-1]  # 将颜色通道的顺序调整为RGB
    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    img_gray_re = img_gray[:, ::-1]  # 测试一下逆转灰度图像的数组

    fig, axe = plt.subplots(nrows=1, ncols=2)
    axe[0].imshow(img_gray, cmap='gray', vmin=0, vmax=255)
    axe[1].imshow(img_gray_re, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()

    fig, axe = plt.subplots(nrows=1, ncols=3)
    axe[0].imshow(img_rgb1)
    axe[1].imshow(img_rgb)
    axe[2].imshow(img_bgr)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(16, 8))
    plt.suptitle("多图显示", fontsize=30)

    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("彩色图像", fontsize=20)
    plt.axis("off")  # 3.11版本的python可以使用False 之前的版本可能只能用"off"

    plt.subplot(1, 3, 2)
    plt.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("灰度图像", fontsize=20)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    # 不指定cmap
    plt.imshow(img_gray, vmin=0, vmax=255)
    plt.title("伪彩色图像", fontsize=20)
    plt.axis("off")

    plt.tight_layout()  # 自动调整子图参数 使之填充整个图像区域
    plt.show()


def pltshow2():
    # figure中num参数的使用
    img_bgr = cv.imread("../Picture/xiangrikui.jpg", cv.IMREAD_COLOR)
    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    plt.figure(1)
    plt.imshow(img_rgb)
    plt.title("rgb彩色图像", fontsize=20)
    plt.axis(False)
    plt.tight_layout()
    plt.show()

    plt.figure(1)
    plt.imshow(img_bgr)
    plt.title("bgr彩色图像", fontsize=20)
    plt.axis(False)
    plt.tight_layout()
    plt.show()

    plt.figure(num=2, figsize=(18, 6))
    plt.suptitle("多图显示", fontsize=30)

    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("RGB图像", fontsize=20)
    plt.axis(False)

    plt.subplot(1, 3, 2)
    plt.imshow(img_bgr)
    plt.title("BGR图像", fontsize=20)
    plt.axis(False)

    plt.tight_layout()
    plt.show()

    # 如果给出的num已经有存在的对应的窗口 则会覆写已存在窗口的内容
    # 已存在的窗口中的内容会清空
    plt.figure(num=2, figsize=(18, 6))

    plt.subplot(1, 3, 3)
    plt.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("灰度图像", fontsize=20)
    plt.axis(False)

    plt.tight_layout()
    plt.show()


def pltshow3():
    # subplots的使用
    img_bgr = cv.imread("../Picture/rose1.jpg", cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
    print(axs)
    axs[0, 0].imshow(img_bgr)
    axs[0, 1].imshow(img_rgb)
    axs[0, 2].imshow(img_gray)
    plt.tight_layout()
    plt.show()

    # 当nrows或者ncols中有取值为1的 axs为一维数组
    fig, axs = plt.subplots(nrows=3, ncols=1)
    # too many indices for array: array is 1-dimensional, but 2 were indexed
    # axs[0, 0].imshow(img_bgr)
    print(axs)
    axs[0].imshow(img_rgb)
    axs[1].imshow(img_gray)
    axs[2].imshow(img_bgr)
    # 这个要每个imshow之后都要进行说明
    plt.axis(False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pltshow1()
    pltshow3()
