import cv2 as cv
import numpy as np


def zh_ch(str):
    # 编码转换
    # 在imshow显示中文总是出现乱码
    # 但是这个函数不能解决问题
    return str.encode('gbk').decode(errors='ignore')


def cvshow():
    # 使用OpenCV读图像
    # opencv读入图像是按照BGR的方式读入 那么显示图像需要进行处理吗
    img = cv.imread("../Picture/cat2.jpg", cv.IMREAD_COLOR)
    if img is None:
        print("无法读取这张图片")
        exit(0)
    cv.namedWindow("cat_bgr", cv.WINDOW_NORMAL | cv.WINDOW_FREERATIO)
    # cv.imshow 在展示图片时 会将三通道逆转 像cv.imread 一样
    cv.imshow("cat_bgr", img)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    cv.namedWindow("cat_rgb", cv.WINDOW_NORMAL | cv.WINDOW_FREERATIO)
    cv.imshow("cat_rgb", img_rgb)
    cv.waitKey(0)
    cv.destroyAllWindows()


def cvwrite():
    # 使用opencv写图像
    # opencv读入图像是按照RGB的方式读入 那么写图像时按照什么方式
    img = cv.imread("../Picture/cat2.jpg", cv.IMREAD_COLOR)
    # cv.imwrite 也会像 cv.imread 一样将三通道进行逆转
    cv.imwrite("../Picture/cat2_bgr.jpg", img)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    cv.imwrite("../Picture/cat2_rgb.jpg", img_rgb)


def cvrws():
    # 使用OpenCV进行图像读写显示
    img = cv.imread('../Picture/xigua1.jpg', cv.IMREAD_COLOR)
    if img is None:
        exit("Could not read the image")
    # 不能显示中文
    cv.imshow("xigua", img)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # [cv.IMWRITE_PNG_STRATEGY, 100] 是什么意思?
    cv.imwrite('../Picture/gray_xigua1.jpg', gray_img, [cv.IMWRITE_PNG_STRATEGY, 100])
    cv.imshow("西瓜灰度图像", gray_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def cvshow1():
    # 在指定窗口中显示图片
    img_bgr = cv.imread("../Picture/rainbow.png", cv.IMREAD_COLOR)
    # 只是创建一个指定窗口 这个函数没有显示图像的功能
    # 相比于imshow 可以设置显示窗口的属性
    cv.namedWindow("rainbow", flags=cv.WINDOW_NORMAL | cv.WINDOW_FREERATIO)
    # 在指定的窗口中显示图像 若不存在 则创建显示窗口
    cv.imshow("rainbow", img_bgr)
    cv.waitKey(0)
    cv.destroyAllWindows()


def cvshow2():
    # 显示多幅图片 进行拼接
    # 由于使用数组拼接 则其维数要相同
    img_bgr = cv.imread("../Picture/rainbow.png", cv.IMREAD_COLOR)
    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    img_r = img_bgr[:, :, 2]
    img_g = img_bgr[:, :, 1]
    img_b = img_bgr[:, :, 0]
    img_2 = np.hstack((img_gray, img_r))
    img_22 = np.hstack((img_b, img_g))
    img_4 = np.vstack((img_2, img_22))
    # rgb 图片
    img_3 = np.dstack((img_r, img_g, img_b))
    # bgr 图片
    img_33 = np.dstack((img_b, img_g, img_r))
    cv.namedWindow("rainbow", flags=cv.WINDOW_NORMAL)
    cv.imshow("rainbow", img_4)
    # 再次使用相同的窗口名称 会覆盖之前的图片
    # 显示RGB图片
    cv.namedWindow("rainbow1", flags=cv.WINDOW_NORMAL)
    cv.imshow("rainbow1", img_3)
    # 显示BGR图片
    cv.namedWindow("rainbow11", flags=cv.WINDOW_NORMAL)
    cv.imshow("rainbow11", img_33)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    cvwrite()
