import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]


def readColorPicture():
    # 读入彩色图像
    img = cv.imread("../Picture/cat1.jpg", cv.IMREAD_COLOR)
    cv.imshow("color picture", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def gray_and_save():
    # 对图像进行灰度化并保存图像
    # 读入彩色图像
    img = cv.imread("../Picture/cat1.jpg", cv.IMREAD_COLOR)
    # 转化为灰度图像
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 保存灰度图像
    cv.imwrite("../Picture/gray_cat1.jpg", img_gray)
    cv.imshow("color picture", img)
    cv.imshow("gray picture", img_gray)
    cv.waitKey(0)
    cv.destroyAllWindows()


def Hist_Equalization(img):
    # 对灰度图像进行直方图均衡化
    L = 256
    pixnumlist = np.zeros(L, np.uint64)  # 像素频数列表
    pixpercentlist = np.zeros(L, np.float32)  # 像素频率列表
    lut = np.zeros(L, np.uint64)  # 像素变换表
    width, height = img.shape
    totalpix = width * height  # 像素总数
    # 像素数量
    for i in range(width):
        for j in range(height):
            pix = img[i, j]
            pixnumlist[pix] = pixnumlist[pix] + 1
    # 像素频率
    p = 0
    for i in range(256):
        pixpercentlist[i] = pixnumlist[i] / totalpix
        p = p + pixpercentlist[i]
    sumpix = 0
    for i in range(256):
        # 概率累积
        sumpix = sumpix + pixpercentlist[i]
        # 计算均衡后的像素
        lut[i] = int((L - 1) * sumpix + 0.5)
    img_ch = lut[img]
    return img_ch


def Hist_Equalization_show():
    # 直方图均衡化测试函数
    # 读入一张较暗的图像
    img = cv.imread("../Picture/dark_cat1.jpg", cv.IMREAD_GRAYSCALE)
    img_ch = Hist_Equalization(img)
    fig, axe = plt.subplots(1, 2, figsize=(12, 5))
    axe[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axe[0].set_title("原图像", fontsize=20)
    axe[0].axis(False)
    axe[1].imshow(img_ch, cmap='gray', vmin=0, vmax=255)
    axe[1].set_title("直方图均衡化后的图像", fontsize=20)
    axe[1].axis(False)
    plt.tight_layout()
    plt.show()


def cal_center(img_part, sigma):
    # 高斯滤波核
    # 3*3模板 计算中心元素滤波后的值
    box = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            # 高斯公式
            box[i, j] = math.exp((-1 / (2 * sigma ** 2)) *
                                 (np.square(i - 1) + np.square(j - 1)))
    box = box / np.sum(box)  # 归一化处理
    return np.clip(int(np.sum(img_part * box) + 0.5), 0, 255)


def gaussian(img, sigma):
    # 利用高斯函数进行滤波处理
    # 边缘填充0
    width, height = img.shape
    img_pad0 = np.zeros((width + 2, height + 2), np.uint8)
    img_solve = np.zeros((width, height), np.uint8)
    for i in range(1, width + 1):
        for j in range(1, height + 1):
            img_pad0[i, j] = img[i - 1, j - 1]
    for i in range(1, width + 1):
        for j in range(1, height + 1):
            # 将中心元素以及周围元素取出
            img_part = np.zeros((3, 3), np.uint8)
            for m in range(3):
                for n in range(3):
                    img_part[m, n] = img_pad0[i + m - 1, j + n - 1]
            # 进行高斯滤波
            img_solve[i - 1, j - 1] = cal_center(img_part, sigma)
    return img_solve


def gaussian_show():
    # 高斯滤波测试函数
    img = cv.imread("../Picture/cat1.jpg", cv.IMREAD_GRAYSCALE)
    sigma = 1.5  # 方差取值为1.5
    img_solve = gaussian(img, sigma)
    fig, axe = plt.subplots(1, 2, figsize=(12, 5))
    axe[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axe[0].set_title("原图像", fontsize=20)
    axe[0].axis(False)
    axe[1].imshow(img_solve, cmap='gray', vmin=0, vmax=255)
    axe[1].set_title("sigma={}, 高斯平滑后的图像".format(sigma), fontsize=20)
    axe[1].axis(False)
    plt.tight_layout()
    plt.show()


def cal_center_b(img_part, d, sigma_s, sigma_c):
    # 双边滤波
    # 3*3模板 计算中心元素滤波后的值
    d_2 = d // 2
    box = np.zeros((d, d))  # 双边滤波核
    for i in range(d):
        for j in range(d):
            if i == d_2 and j == d_2:
                continue
            # 双边滤波核
            box[i, j] = math.exp((-1 / (2 * sigma_s ** 2)) * (np.square(i - d_2) + np.square(j - d_2)) +
                                 (-1 / (2 * sigma_c ** 2)) * np.square(img_part[d_2, d_2] - img_part[i, j]))
    box[d_2, d_2] = 0  # 中心元素不做处理
    box = box / np.sum(box)  # 归一化处理
    return np.clip(int(np.sum(img_part * box) / np.sum(box) + 0.5), 0, 255)


def bilateral(img, d, sigma_s, sigma_c):
    # 对灰度图像 单颜色分量图像
    # 双边滤波
    # 边缘填充0
    d_2 = d // 2
    width, height = img.shape
    img_pad0 = np.zeros((width + 2 * d_2, height + 2 * d_2), np.uint8)
    img_solve = np.zeros((width, height), np.uint8)
    for i in range(d_2, width + d_2):
        for j in range(d_2, height + d_2):
            img_pad0[i, j] = img[i - d_2, j - d_2]
    for i in range(d_2, width + d_2):
        for j in range(d_2, height + d_2):
            # 将中心元素以及周围元素取出
            img_part = np.zeros((d, d), np.uint8)
            for m in range(d):
                for n in range(d):
                    img_part[m, n] = img_pad0[i + m - d_2, j + n - d_2]
            # 双边滤波
            img_solve[i - d_2, j - d_2] = cal_center_b(img_part, d, sigma_s, sigma_c)
    return img_solve


def bilateral_show():
    # 双边滤波测试函数
    # 读入彩色图像
    img_bgr = cv.imread("../Picture/cat1.jpg", cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    d = 5
    sigma_s = 20
    sigma_c = 20
    img_r = bilateral(img_rgb[:, :, 0], d, sigma_s, sigma_c)
    img_g = bilateral(img_rgb[:, :, 1], d, sigma_s, sigma_c)
    img_b = bilateral(img_rgb[:, :, 2], d, sigma_s, sigma_c)
    img_solve = np.dstack((img_r, img_g, img_b))
    fig, axe = plt.subplots(1, 2, figsize=(12, 5))
    axe[0].imshow(img_rgb)
    axe[0].set_title("原图像", fontsize=20)
    axe[0].axis(False)
    axe[1].imshow(img_solve)
    axe[1].set_title("d={0} sigma_s={1} sigma_c={2},双边滤波后的图像".format(d, sigma_s, sigma_c), fontsize=20)
    axe[1].axis(False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    readColorPicture()
    gray_and_save()
    Hist_Equalization_show()
    gaussian_show()
    bilateral_show()
