import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']


def bitogray(img_bi):
    img = img_bi.copy().astype(np.uint8)
    for i in range(len(img_bi)):
        imgl = []
        for j in range(len(img_bi[0])):
            if img_bi[i][j] is True:
                imgl.append(255)
            else:
                imgl.append(0)
        img.append(imgl)

    return img


def basesegment():
    img0 = cv.imread("../Picture/cat1.jpg", cv.IMREAD_GRAYSCALE)
    img = img0.astype(np.float32)
    count = 0
    Delta_T = 0.5
    Thr_new = 0
    Thr = np.mean(img)
    done = True
    while done:
        count = count + 1
        Thr_new = 0.5 * (np.mean(img[img > Thr]) + np.mean(img[img <= Thr]))
        if np.abs(Thr - Thr_new) < Delta_T:
            done = False
        Thr = Thr_new
    img_bw = img > Thr
    print('阈值=', Thr)
    print("迭代次数=", count)

    plt.figure(figsize=(12, 6))
    plt.suptitle("基本全局阈值图像分割", fontdict={'size': 15})
    plt.gray()

    plt.subplot(1, 2, 1)
    plt.imshow(img0, vmin=0, vmax=255)
    plt.title("原图像", fontdict={'size': 15})
    plt.axis(False)

    plt.subplot(1, 2, 2)
    img_bw.astype(np.uint8)
    plt.imshow(img_bw)
    plt.title("阈值分割后的图像", fontdict={'size': 15})
    plt.axis(False)

    plt.show()


if __name__ == "__main__":
    basesegment()
