import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]


def get_pk_mk(img):
    # 获取像素频率累加列表pk 像素灰度平均值列表mk
    L = 256
    width, height = img.shape
    totalnum = width * height
    pixnumlist = np.zeros(L, np.uint64)  # 像素频数列表
    pk = np.zeros(L, np.float32)  # 像素频率累加列表
    mk = np.zeros(L, np.float32)  # 像素灰度平均值列表
    p = 0
    m = 0
    for i in range(width):  # 像素数量
        for j in range(height):
            pix = img[i, j]
            pixnumlist[pix] = pixnumlist[pix] + 1
    for k in range(L):
        percent = pixnumlist[k] / totalnum
        p = p + percent
        pk[k] = p
        m = m + k * percent
        mk[k] = m
    return pk, mk


def get_thresh(img):
    # 获取图像阈值
    L = 256
    pk, mk = get_pk_mk(img)
    maxk = 0
    maxrou_square = 0
    for k in range(L):
        p1 = pk[k]
        p2 = pk[L - 1] - p1
        if p1 == 0 or p2 == 0:
            rou_square = 0
        else:
            # 根据公式进行计算
            m1 = 1 / p1 * mk[k]
            m2 = 1 / p2 * (mk[L - 1] - mk[k])
            rou_square = (p1 * (m1 - mk[L - 1]) * (m1 - mk[L - 1])
                          + p2 * (m2 - mk[L - 1]) * (m2 - mk[L - 1]))
        if rou_square > maxrou_square:
            maxrou_square = rou_square
            maxk = k
    return maxk


def binarization(img, T):
    # 根据阈值T 将图像进行二值化
    img_change = img.copy()
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            if img[i, j] > T:  # 大于阈值 置255
                img_change[i, j] = 255
            else:
                img_change[i, j] = 0
    return img_change


def binarization_show():
    # 图像二值化 阈值分割测试函数
    img = cv.imread("../Picture/cat1.jpg", cv.IMREAD_GRAYSCALE)
    T = get_thresh(img)
    img_bi = binarization(img, T)
    fig, axe = plt.subplots(1, 2, figsize=(12, 5))
    axe[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axe[0].set_title("原图像", fontsize=20)
    axe[0].axis(False)
    axe[1].imshow(img_bi, cmap='gray', vmin=0, vmax=255)
    axe[1].set_title("阈值分割后的图像，阈值T={}".format(T), fontsize=20)
    axe[1].axis(False)
    plt.tight_layout()
    plt.show()


def cal_center(img_part):
    # 使用sobel算子计算中心像素的值
    gx = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    gy = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    pix_c = np.sqrt(np.square(np.sum(img_part * gx)) + np.square(np.sum(img_part * gy)))
    return np.clip(int(pix_c + 0.5), 0, 255)


def getedge_sobel(img):
    # 利用sobel算子检测图像边缘
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
            # 使用sobel算子计算中心像素的值
            img_solve[i - 1, j - 1] = cal_center(img_part)
    return img_solve


def sobel_show():
    # 使用sobel算子计算测试函数
    img = cv.imread("../Picture/cat1.jpg", cv.IMREAD_GRAYSCALE)
    img_solve = getedge_sobel(img)
    fig, axe = plt.subplots(1, 2, figsize=(12, 5))
    axe[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axe[0].set_title("原图像", fontsize=20)
    axe[0].axis(False)
    axe[1].imshow(img_solve, cmap='gray', vmin=0, vmax=255)
    axe[1].set_title("sobel边缘图像", fontsize=20)
    axe[1].axis(False)
    plt.tight_layout()
    plt.show()


def line_picture():
    # 生成包含直线的二值图像
    img = np.zeros((100, 150), dtype=np.uint8)
    img = cv.line(img, (65, 0), (65, 99), 255, 1)
    img = cv.line(img, (0, 30), (150, 30), 255, 1)
    img = cv.line(img, (130, 60), (10, 80), 255, 1)
    img = cv.line(img, (20, 40), (100, 80), 255, 1)
    img = cv.line(img, (100, 5), (135, 95), 255, 1)
    return img


def hough(img):
    # hough变换
    lines = cv.HoughLines(img, rho=1, theta=np.pi / 90, threshold=20)
    print("距离 rho:", lines[:, 0, 0])
    print("角度 theta:", np.round(np.rad2deg(lines[:, 0, 1])))
    # 将灰度图像转换为RGB颜色通道图像，用黄色绘出检测到的每根直线
    imgresult = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    # 设定检测到的直线x轴坐标范围
    xp = np.arange(0, img.shape[1])
    # 根据参数rho，theta计算每条直线对应的y轴坐标值
    for line in lines:
        rho, theta = line[0]
        if theta == 0:  # 画垂直直线
            x1 = np.int32(rho)  # 确定在图像范围内的直线端点坐标
            y1 = 0
            x2 = np.int32(rho)
            y2 = img.shape[0] - 1
        else:
            yp = np.int32((rho - xp * np.cos(theta)) / np.sin(theta))
            yidx = np.logical_and(yp >= 0, yp < img.shape[0])
            x1 = xp[yidx][0]
            y1 = yp[yidx][0]
            x2 = xp[yidx][-1]
            y2 = yp[yidx][-1]
        # 画线
        imgresult = cv.line(imgresult, (x1, y1), (x2, y2), (255, 255, 0), 1)
    return imgresult


def hough_show():
    # hough变换测试函数
    img = line_picture()
    img_result = hough(img)
    fig, axe = plt.subplots(1, 2, figsize=(12, 5))
    axe[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axe[0].set_title("原图像", fontsize=20)
    axe[0].axis(False)
    axe[1].imshow(img_result, cmap='gray', vmin=0, vmax=255)
    axe[1].set_title("直线检测", fontsize=20)
    axe[1].axis(False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    binarization_show()
    sobel_show()
    hough_show()
