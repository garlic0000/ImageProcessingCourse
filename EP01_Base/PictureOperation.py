from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def manual():
    img = cv.imread('../Picture/lingxiaohua.jpg')
    cv.imshow("img", img)
    img1 = img.astype(np.float16) + 50
    img1 = np.clip(img1, 0, 255).astype(np.uint8)  # 饱和处理
    cv.imshow("img1", img1)
    img2 = img.astype(np.float16) + [50, 50, 0]
    img2 = np.clip(img2, 0, 255).astype(np.uint8)
    cv.imshow("img2", img2)
    img3 = img.astype(np.float16) * 1.5
    img3 = np.clip(img3, 0, 255).astype(np.uint8)
    cv.imshow("img3", img3)
    cv.waitKey(0)
    cv.destroyAllWindows()


def calwithcv():
    img = cv.imread("../Picture/lingxiaohua.jpg")
    cv.imshow("img", img)
    img1 = cv.add(img, 50)
    cv.imshow("img1", img1)
    # 四元组可以 第四个好像没什么意义 只有前3个值
    img2 = cv.add(img, (40, 0, 0, 50))
    # 无法使用三元组 或 3个元素的列表
    # img2 = cv.add(img, [50, 50, 0])
    # img2 = cv.add(img, (50, 50, 0))
    cv.imshow("img2", img2)
    img3 = cv.multiply(img, 1.5)
    cv.imshow("img3", img3)
    cv.waitKey(0)
    cv.destroyAllWindows()


def picturelogical():
    img1 = cv.imread("../Picture/cat1.jpg")
    img2 = cv.imread("../Picture/gray_cat1.png")
    print(np.logical_and(img1, img2))


def arraycp():
    f = np.array([[125, 36], [79, 66]])
    g = np.array([[100, 36], [88, 0]])
    x = f < g
    print('x=', x)
    y = f > 80
    print('y=', y)
    print(np.greater(f, g))
    print(np.less_equal(f, g))


def picturecp():
    img1 = cv.imread("../Picture/cat1.jpg")
    img2 = cv.imread("../Picture/gray_cat1.png")
    print(np.greater_equal(img1, img2))


def picturebitnp():
    img1 = cv.imread("../Picture/cat1.jpg")
    img2 = cv.imread("../Picture/gray_cat1.png")
    print(np.bitwise_xor(img1, img2))
    img3 = np.bitwise_not(img1)
    cv.imshow("img1", img1)
    cv.imshow("img3", img3)
    cv.waitKey(0)
    cv.destroyAllWindows()


def picturebitcv():
    img1 = cv.imread("../Picture/lingxiaohua.jpg")
    img2 = cv.imread("../Picture/gray_lingxiaohua.png")
    print(cv.bitwise_or(img1, img2))
    img3 = cv.bitwise_not(img1)
    cv.imshow("img1", img1)
    cv.imshow("img3", img3)
    cv.waitKey(0)
    cv.destroyAllWindows()


def overflowcheck():
    # 溢出检测
    x = np.array([252, 236, 211], dtype=np.uint8)
    # np.uint8 在0~255之内 大于255 小于0 会溢出 相当于取余操作
    y = x + 20
    print(y, y.dtype)
    z = x - 255
    print(z, z.dtype)


def saturatesolve():
    # 饱和处理
    x = np.array([252, 236, 211], dtype=np.uint8)
    y = x.astype(np.float16) + 20
    print(y, y.dtype)
    y = np.clip(y, 0, 255).astype(np.uint8)
    print(y, y.dtype)
    z = x.astype(np.float16) - 255
    print(z, z.dtype)
    z = np.clip(z, 0, 255).astype(np.uint8)
    print(z, z.dtype)


if __name__ == "__main__":
    manual()
    calwithcv()
