import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, feature, util, color
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]


def harris(img):
    # 角点检测
    harris_res = feature.corner_harris(img)  # 计算Harris角点响应函数
    # 从Harris角点响应图像获取峰值点及其行列下标，即为角点
    coords = feature.corner_peaks(harris_res, min_distance=5, threshold_rel=0.01)
    img_corners = util.img_as_ubyte(color.gray2rgb(img))
    for corner in coords:
        # 画红色圆环
        img_corners = cv.circle(img_corners, (corner[1], corner[0]), 12, (255, 0, 0), thickness=2)
    return img_corners


def harris_show():
    # 角点检测测试函数
    img = io.imread("../Picture/xiangqipan.png")
    img_corners = harris(img)
    fig, axe = plt.subplots(1, 2, figsize=(12, 5))
    axe[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axe[0].set_title("原图像", fontsize=20)
    axe[0].axis(False)
    axe[1].imshow(img_corners, cmap='gray', vmin=0, vmax=255)
    axe[1].set_title("角点检测图像", fontsize=20)
    axe[1].axis(False)
    plt.tight_layout()
    plt.show()


def cal_center(img_part):
    # 计算每个像素的LBP特征值
    center = 0
    pos = 0
    for i in range(3):  # 第一行
        if img_part[0, i] > img_part[1, 1]:
            bi = 1
        else:
            bi = 0
        center = center + bi * 2 ** pos
        pos = pos + 1
    if img_part[1, 2] > img_part[1, 1]:  # 中间最右
        bi = 1
    else:
        bi = 0
    center = center + bi * 2 ** pos
    for i in range(2, -1, -1):  # 第三行
        # 第一行
        if img_part[2, i] > img_part[1, 1]:
            bi = 1
        else:
            bi = 0
        center = center + bi * 2 ** pos
        pos = pos + 1
    if img_part[1, 0] > img_part[1, 1]:  # 中间最左
        bi = 1
    else:
        bi = 0
    center = center + bi * 2 ** pos
    return center


def lbp(img):
    # 生成LBP特征图像
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
            # 计算每个像素的LBP特征
            img_solve[i - 1, j - 1] = cal_center(img_part)
    return img_solve


def lbp_show():
    # LBP特征图像测试函数
    img = cv.imread("../Picture/cat1.jpg", cv.IMREAD_GRAYSCALE)
    img_lbp = lbp(img)
    print("原图像每个像素的LBP特征:", img_lbp)
    fig, axe = plt.subplots(1, 2, figsize=(12, 5))
    axe[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axe[0].set_title("原图像", fontsize=20)
    axe[0].axis(False)
    axe[1].imshow(img_lbp, cmap='gray', vmin=0, vmax=255)
    axe[1].set_title("图像的LBP特征", fontsize=20)
    axe[1].axis(False)
    plt.tight_layout()
    plt.show()


def square_differ_match(img, template):
    # 平方差匹配法
    th, tw = template.shape[:2]  # 获取图像的高和宽
    result = cv.matchTemplate(img, template, cv.TM_SQDIFF)  # 平方差匹配法
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    tl = min_loc  # 匹配数值越低 效果越好
    br = (tl[0] + tw, tl[1] + th)
    img_c = img.copy()
    cv.rectangle(img_c, tl, br, (0, 0, 255), 2)  # 在图上绘制矩形
    img_c = cv.cvtColor(img_c, cv.COLOR_BGR2RGB)
    return img_c


def normalize_related_match(img, template):
    # 归一化相关匹配法
    th, tw = template.shape[:2]  # 获取图像的高和宽
    result = cv.matchTemplate(img, template, cv.TM_CCORR_NORMED)  # 归一化相关匹配法
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    tl = max_loc  # 匹配数值越高 效果越好
    br = (tl[0] + tw, tl[1] + th)
    img_c = img.copy()
    cv.rectangle(img_c, tl, br, (0, 0, 255), 2)  # 在图上绘制矩形
    img_c = cv.cvtColor(img_c, cv.COLOR_BGR2RGB)
    return img_c


def match_show():
    # 模板匹配测试函数
    img_bgr = cv.imread("../Picture/chepai.jpg", cv.IMREAD_COLOR)
    template_bgr = cv.imread("../Picture/word.jpg", cv.IMREAD_COLOR)
    img = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    template = cv.cvtColor(template_bgr, cv.COLOR_BGR2RGB)
    res_sd = square_differ_match(img_bgr, template_bgr)  # 平方差匹配法
    res_nr = normalize_related_match(img_bgr, template_bgr)  # 归一化相关匹配法
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title("原图", fontsize=20)
    plt.axis(False)
    plt.subplot(2, 2, 2)
    plt.imshow(template)
    plt.title("进行匹配的图像", fontsize=20)
    plt.axis(False)
    plt.subplot(2, 2, 3)
    plt.imshow(res_sd)
    plt.title("平方差匹配法匹配结果", fontsize=20)
    plt.axis(False)
    plt.subplot(2, 2, 4)
    plt.imshow(res_nr)
    plt.title("归一化相关匹配法匹配结果", fontsize=20)
    plt.axis(False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    harris_show()
    lbp_show()
    match_show()
