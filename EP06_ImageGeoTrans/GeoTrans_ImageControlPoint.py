import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]


def rechange():
    img = cv.imread("../Picture/niuqu3.png", cv.IMREAD_GRAYSCALE)
    src = np.float32([[155, 15], [65, 40], [260, 130], [360, 95]])
    dst = np.float32([[0, 0], [0, 50], [300, 50], [300, 0]])
    Mat_pe = cv.getPerspectiveTransform(src, dst)
    img_pe = cv.warpPerspective(img, Mat_pe, (300, 50))

    imgs = [img, img_pe]
    imgnames = ["原图像", "矫正后的图像"]
    fig, axe = plt.subplots(nrows=1, ncols=2)
    for i in range(len(imgs)):
        axe[i].imshow(imgs[i], cmap='gray', vmin=0, vmax=255)
        axe[i].set_title(imgnames[i], fontsize=10)
        axe[i].axis(False)
    plt.tight_layout()
    plt.show()


def readQR():
    img = cv.imread("../Picture/qrcode3.png", cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    qrcode_detector = cv.QRCodeDetector()
    data, vertices, rectified_qrcode_binarized = qrcode_detector.detectAndDecode(img)
    if len(data) > 0:
        print("解码信息:{}".format(data))
        pts = np.int32(vertices).reshape(-1, 2)
        img = cv.polylines(img, [pts], True, (0, 255, 0), 5)
        for j in range(pts.shape[0]):
            cv.circle(img, (pts[j, 0], pts[j, 1]), 10, (0, 0, 255), -1)
        imgs = [img, np.uint8(rectified_qrcode_binarized)]
        imgnames = ["解码数据:{}".format(data), "QRcode"]
        fig, axe = plt.subplots(nrows=1, ncols=2)
        for i in range(len(imgs)):
            axe[i].imshow(imgs[i])
            axe[i].set_title(imgnames[i], fontsize=10)
            axe[i].axis(False)
        plt.tight_layout()
        plt.show()
    else:
        print("没有检测到数据")


if __name__ == "__main__":
    # rechange()
    readQR()
