import cv2 as cv
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]


def color2gray_cv():
    img = cv.imread("../Picture/lingxiaohua.jpg")
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    plt.figure(figsize=(12, 6))
    plt.suptitle("彩色图像转灰度图像", fontdict={'size': 20})

    plt.subplot(1, 2, 1)
    plt.imshow(img[:, :, ::-1])
    plt.title("彩色图像", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 2, 2)
    plt.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("灰度图像", fontdict={'size': 15})
    plt.axis(False)

    plt.show()


if __name__ == "__main__":
    color2gray_cv()
