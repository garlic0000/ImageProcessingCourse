import cv2 as cv
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]


def scale_transf_cv():
    # 使用opencv进行图像缩放
    # 330 279
    img_bgr = cv.imread("../Picture/tomato1.jpg", cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    img_big = cv.resize(img_rgb, dsize=(0, 0), fx=1.5, fy=1.5)
    width = 165
    height = 139
    img_small = cv.resize(img_rgb, dsize=(width, height), interpolation=cv.INTER_NEAREST)
    # cv.namedWindow("big", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    # cv.imshow("big", img_big)
    # cv.namedWindow("small", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    # cv.imshow("small", img_small)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    imgs = [img_rgb, img_big, img_small]
    imgnames = ["原图像", "放大1.5倍", "缩小0.5倍"]
    fig, axe = plt.subplots(nrows=1, ncols=3)
    for i in range(len(imgs)):
        axe[i].imshow(imgs[i])
        axe[i].set_title(imgnames[i], fontsize=10)
        axe[i].axis(False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    scale_transf_cv()
