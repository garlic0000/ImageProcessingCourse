import cv2 as cv
import numpy as np


def colortrans_cv():
    img = cv.imread('../Picture/lingxiaohua.jpg', cv.IMREAD_COLOR)
    imghls = cv.cvtColor(img.astype(np.float32)/255, cv.COLOR_BGR2HLS)
    imghsv = cv.cvtColor(img.astype(np.float32)/255, cv.COLOR_BGR2HSV)
