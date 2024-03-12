import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage import io, util
# 设置中文字符
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]


def pwline_rescale_intensity(image, r1, s1, r2, s2):
    if np.logical_or(r1 >= r2, s1 >= s2):
        print("the control point is invalid")
        exit()
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if i < r1:
            lut[i] = i * s1 / r1
        elif i <= r2:
            lut[i] = (s2 - s1) / (r2 - r1) * (i - r1) + s1
        else:
            lut[i] = (255 - s2) / (255 - r2) * (i - r2) + s2
    img_out = lut[image]
    return img_out, lut


def drawlinegraph(x_vector, y_vector, title, rows, cols, idx):
    plt.subplot(rows, cols, idx)
    plt.title(title, fontdict={'size': 30})
    plt.plot(x_vector, y_vector)
    plt.scatter(x_vector, y_vector, c="green")
    plt.xticks(np.arange(0, 251, 50), fontsize=25)
    plt.yticks(np.arange(0, 251, 50), fontsize=25)
    plt.xlabel("输入灰度值", fontdict={'size': 25})
    plt.ylabel("输出灰度值", fontdict={'size': 25})


def showpicture(img, title, rows, cols, idx):
    plt.subplot(rows, cols, idx)
    plt.imshow(img)
    plt.title(title, fontdict={'size': 30})
    plt.axis(False)


def drawhist(img, title, rows, cols, idx):
    plt.subplot(rows, cols, idx)
    plt.title(title, fontdict={'size': 30})
    plt.hist(img.ravel(), 256, histtype='bar')
    plt.xticks(np.arange(0, 256, step=50), fontsize=25)
    plt.yticks(fontsize=25)


def contraststretch1():
    img = io.imread("../Picture/lingxiaohua.jpg", as_gray=True)
    # 返回值为小数 需要转换
    img = util.img_as_ubyte(img)
    r1 = 80
    s1 = 10
    r2 = 150
    s2 = 220
    img_pw, lut = pwline_rescale_intensity(img, r1, s1, r2, s2)
    plt.figure(figsize=(24, 8))
    showpicture(img, "凌霄花", 1, 3, 1)
    showpicture(img, "凌霄花_分段线性变化后", 1, 3, 2)
    drawlinegraph((0, r1, r2, 255), (0, s1, s2, 255), "灰度变化曲线", 1, 3, 3)
    plt.show()


def contraststrentch2():
    # img1 = io.imread("../Picture/lingxiaohua.jpg", as_gray=True)
    # img1 = util.img_as_ubyte(img1)
    # img2 = io.imread("../Picture/bright_lingxiaohua.jpg", as_gray=True)
    # img2 = util.img_as_ubyte(img2)
    # img3 = io.imread("../Picture/dark_lingxiaohua.jpg", as_gray=True)
    # img3 = util.img_as_ubyte(img3)
    img1 = cv.imread("../Picture/lingxiaohua.jpg", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("../Picture/bright_lingxiaohua.jpg", cv.IMREAD_GRAYSCALE)
    img3 = cv.imread("../Picture/dark_lingxiaohua.jpg", cv.IMREAD_GRAYSCALE)

    r1_img1 = np.min(img1)
    r2_img1 = np.max(img1)
    r1_img2 = np.min(img2)
    r2_img2 = np.max(img2)
    r1_img3 = np.min(img3)
    r2_img3 = np.max(img3)
    s1 = 0
    s2 = 255

    img1_pw, lut1 = pwline_rescale_intensity(img1, r1_img1, s1, r2_img1, s2)
    img2_pw, lut2 = pwline_rescale_intensity(img2, r1_img2, s1, r2_img2, s2)
    img3_pw, lut3 = pwline_rescale_intensity(img3, r1_img3, s1, r2_img3, s2)

    plt.figure(figsize=(40, 30))

    showpicture(img1, "凌霄花", 3, 4, 1)
    showpicture(img1_pw, "处理后的凌霄花", 3, 4, 2)
    drawlinegraph((0, r1_img1, r2_img1, 255), (0, s1, s2, 255), "灰度变化曲线", 3, 4, 3)
    drawhist(img1_pw, "灰度直方图", 3, 4, 4)

    showpicture(img2, "凌霄花", 3, 4, 5)
    showpicture(img2_pw, "处理后的凌霄花", 3, 4, 6)
    drawlinegraph((0, r1_img2, r2_img2, 255), (0, s1, s2, 255), "灰度变化曲线", 3, 4, 7)
    drawhist(img2_pw, "灰度直方图", 3, 4, 8)

    showpicture(img3, "凌霄花", 3, 4, 9)
    showpicture(img3_pw, "处理后的凌霄花", 3, 4, 10)
    drawlinegraph((0, r1_img3, r2_img3, 255), (0, s1, s2, 255), "灰度变化曲线", 3, 4, 11)
    drawhist(img3_pw, "灰度直方图", 3, 4, 12)

    plt.show()


if __name__ == "__main__":
    contraststretch1()
    contraststrentch2()
