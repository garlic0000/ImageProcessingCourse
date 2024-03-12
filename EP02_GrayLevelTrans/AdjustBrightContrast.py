import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]


def adjust_bright_contrast(image, b, c):
    # b c 的取值范围为 [0, 1]
    k = np.tan((45 + 44 * c)*np.pi/180)
    lookUpTable = np.zeros((1, 256), np.uint8)
    for i in range(256):
        s = (i - 127.5*(1-b))*k + 127.5*(1+b)
        lookUpTable[0, i] = np.clip(s, 0, 255)
    img_out = cv.LUT(image, lookUpTable)
    return img_out


def showimage():
    img = cv.imread("../Picture/lingxiaohua.jpg", cv.IMREAD_GRAYSCALE)
    img_out = adjust_bright_contrast(img, -0.35, 0.4)
    plt.figure(figsize=(12, 6))
    plt.gray()
    plt.subplot(1, 2, 1)
    plt.imshow(img, vmin=0, vmax=255)
    plt.title("凌霄花")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img_out, vmin=0, vmax=255)
    plt.title("调整后的凌霄花")
    plt.axis("off")
    plt.show()

def cvchange_bright():
    img = cv.imread("../Picture/cat1.jpg")
    img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(img_RGB)
    plt.title("img")
    img1 = cv.add(img_RGB, 100)
    cv.imwrite("../Picture/bright_cat.jpg", img1)
    plt.show()

    plt.figure()
    plt.imshow(img1)
    plt.title("img1")
    img2 = cv.subtract(img_RGB, 100)
    plt.show()

    cv.imwrite("../Picture/dark_cat.jpg", img2)
    plt.figure()
    plt.imshow(img2)
    plt.title("img2")
    plt.show()


def npchange_bright():
    img = cv.imread("../Picture/lingxiaohua.jpg")
    img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    img1 = img_RGB.astype(np.float16) + 100
    img1 = np.clip(img1, 0, 255).astype(np.uint8)
    cv.imwrite("../Picture/bright_lingxiaohua.jpg", img1)

    img2 = img_RGB.astype(np.float16) - 100
    img2 = np.clip(img2, 0, 255).astype(np.uint8)
    cv.imwrite("../Picture/dark_lingxiaohua.jpg", img2)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img_RGB)
    plt.title("img_RBG")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img1)
    plt.title("img1")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img2)
    plt.title("img2")
    plt.axis("off")

    plt.show()

def cvchange_contrast():
    img = cv.imread("../Picture/cat1.jpg")
    img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img1 = cv.multiply(img_RGB, 1.5)
    img2 = cv.multiply(img_RGB, 0.5)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img_RGB)
    plt.title("img_RGB")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img1)
    plt.title("img1")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img2)
    plt.title("img2")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    showimage()






    
