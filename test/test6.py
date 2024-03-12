from skimage import io, util
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]


def regionGrow(grayimage, seeds, thresh, neighbors):
    gray = util.img_as_float(grayimage.copy())
    seedMark = np.zeros(grayimage.shape).astype(np.uint8)
    connection = np.zeros(8)
    if neighbors == 8:
        connection = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                      (1, 1), (1, 0), (1, -1), (0, -1)]
    elif neighbors == 4:
        connection = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    numpixels = 1.0
    growed_region_mean = gray[seeds[0][0], seeds[0][1]]
    growed_region_sum = growed_region_mean
    while len(seeds) != 0:
        pt = seeds.pop(0)
        for i in range(neighbors):
            tmpX = pt[0] + connection[i][0]
            tmpY = pt[1] + connection[i][1]
            if tmpX < 0 or tmpY < 0 or tmpX >= gray.shape[0] or tmpY >= gray.shape[1]:
                continue
            gray_diff = abs(gray[tmpX, tmpY] - growed_region_mean)
            if (gray_diff < thresh) and (seedMark[tmpX, tmpY] == 0):
                seedMark[tmpX, tmpY] = 255
                seeds.append((tmpX, tmpY))
                growed_region_sum += gray[tmpX, tmpY]
                numpixels += 1
                growed_region_mean = growed_region_sum / numpixels
    return seedMark


# def show(imgs, rows, cols):
#     fig, axe = plt.subplots(nrows=rows, ncols=cols)


def main():
    # img = io.imread("../Picture/cat1.jpg", cv.IMREAD_GRAYSCALE)
    img = cv.imread("../Picture/cat1.jpg", cv.IMREAD_GRAYSCALE)
    seeds = [(125, 250)]
    seedMark1 = regionGrow(img, seeds, thresh=0.06, neighbors=8)
    seeds = [(125, 250)]
    seedMark2 = regionGrow(img, seeds, thresh=0.2, neighbors=8)
    img_res1 = img.copy()
    img_res1[seedMark1 > 0] = 255
    img_res2 = img.copy()
    img_res2[seedMark2 > 0] = 255

    imgs = [img, img_res1, img_res2]
    imgtitles = ["原图像", "thresh=0.06", "thresh=0.2"]
    fig, axe = plt.subplots(nrows=1, ncols=3)
    for i in range(3):
        axe[i].imshow(imgs[i], cmap='gray', vmin=0, vmax=255)
        axe[i].set_title(imgtitles[i], fontsize=10)
        axe[i].axis(False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
