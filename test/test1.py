import cv2 as cv


# 1.利用某种高级程序设计语言(如：C++/OpenCV、Python)读取彩色图像文件。
def readcolorimage(imgpath):
    img = cv.imread(imgpath, cv.IMREAD_COLOR)
    cv.imshow("彩色图像", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
