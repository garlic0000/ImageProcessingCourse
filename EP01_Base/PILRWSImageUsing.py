import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import numpy as np
from skimage import data, util
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]


def pilrws():
    img = Image.open("../Picture/rainbow.png")
    img_gray = img.convert('L')
    img_rgb = img.convert('RGB')
    img_rgb.save('../Picture/rainbow.jpg')
    print(img.format, img.size, img.mode)
    # 调动系统的图像显示程序
    img.show("原图")
    img_gray.show("灰度图像")
    img_rgb.show("rgb图像")


def pilrw():
    chelsea = data.chelsea()
    # 将数组的三通道逆过来存放
    # cv.imread 读图像时 是按照BGR的方式读入 cv.write 会将图像三通道逆转后存放
    # chelsea = cv.cvtColor(chelsea, cv.COLOR_BGR2RGB)
    chelsea = chelsea[:, :, ::-1]
    cv.imwrite("../Picture/chelsea.jpg", chelsea)
    # Image.open 不能打开数组 要打开路径
    img = Image.open("../Picture/chelsea.jpg")
    img.show()


def pilrw1():

    img = Image.open("../Picture/rainbow.png")
    img_gray = img.convert('L')
    img_gray = np.asarray(img_gray)
    img_rgb = img.convert('RGB')
    img_rgb = np.asarray(img_rgb)
    # 二值图像
    img_1 = img.convert("1")
    img_1 = np.asarray(img_1)
    img = np.asarray(img)
    imgs = [img, img_gray, img_rgb, img_1]
    fig, axe = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    pos = 0
    for i in range(2):
        for j in range(2):
            axe[i, j].imshow(imgs[pos], cmap='gray', vmin=0, vmax=255)
            axe[i, j].axis(False)
            pos = pos + 1
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pilrw1()
