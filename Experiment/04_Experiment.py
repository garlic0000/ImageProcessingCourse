import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, feature, util, color
from scipy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]


def cal_fourier_frequency(img):
    # 计算灰度图像的傅里叶频谱
    M, N = img.shape
    P = cv.getOptimalDFTSize(M)
    Q = cv.getOptimalDFTSize(N)
    # 在原图像下bottom、右right，按指定方式扩展P-M行、Q-N列
    imgn = cv.copyMakeBorder(img, top=0, bottom=P - M, left=0, right=Q - N, borderType=cv.BORDER_REPLICATE)
    # 计算扩展图像的DFT
    imgdft = cv.dft(np.float32(imgn), flags=cv.DFT_COMPLEX_OUTPUT)
    # 计算图像的幅度谱
    imgdft_mag = cv.magnitude(imgdft[:, :, 0], imgdft[:, :, 1])
    imgdft_cent = fftshift(imgdft)  #频谱中心化
    # 计算中心化幅度谱
    imgdft_mag_cent = cv.magnitude(imgdft_cent[:, :, 0], imgdft_cent[:, :, 1])
    # 为便于观察图像的幅度谱，采用对数校正增强幅度谱图像
    c1 = 255 / np.log10(1 + np.max(imgdft_mag))
    imgdft_mag_log = c1 * np.log10(1 + imgdft_mag)
    c2 = 255 / np.log10(1 + np.max(imgdft_mag_cent))
    imgdft_mag_cent_log = c2 * np.log10(1 + imgdft_mag_cent)
    return imgdft_mag_log, imgdft_mag_cent_log


def fourier_show():
    # 傅里叶变换测试函数
    img = cv.imread("../Picture/cat1.jpg", cv.IMREAD_GRAYSCALE)
    imgdft_mag_log, imgdft_mag_cent_log = cal_fourier_frequency(img)
    plt.figure(figsize=(18, 5))
    plt.gray()
    plt.subplot(1, 3, 1)
    plt.imshow(img, vmin=0, vmax=255)
    plt.title("原图像", fontsize=20)
    plt.axis(False)
    plt.subplot(1, 3, 2)
    plt.imshow(imgdft_mag_log, vmin=0, vmax=255)
    plt.title("未中心化幅度谱", fontsize=20)
    plt.axis(False)
    plt.subplot(1, 3, 3)
    plt.imshow(imgdft_mag_cent_log, vmin=0, vmax=255)
    plt.title("中心化幅度谱", fontsize=20)
    plt.axis(False)
    plt.tight_layout()
    plt.show()


def noise_img(img, var):
    # 方差为var
    # 加入高斯白噪声 生成噪声图像
    img_n = util.random_noise(img, mode='gaussian', var=var)
    img_n = util.img_as_ubyte(img_n)
    return img_n


def butterworth_low(img, n, D0):
    # 巴特沃斯低通处理函数 彩色图像
    M, N = img.shape[0:2]
    imgex = np.pad(img, ((0, M), (0, N), (0, 0)), mode='reflect')
    Fp = fft2(imgex, axes=(0, 1))  # 计算图像的DFT,并中心化
    Fp = fftshift(Fp, axes=(0, 1))
    v = np.arange(-N, N)  # 构建平面坐标网格
    u = np.arange(-M, M)
    Va, Ua = np.meshgrid(v, u)
    Da2 = Ua ** 2 + Va ** 2
    HBlpf = 1 / (1 + (Da2 / D0 ** 2) ** n)
    HBlpf_3D = np.dstack((HBlpf, HBlpf, HBlpf))
    Gp = Fp * HBlpf_3D  # 计算传递函数
    Gp = ifftshift(Gp, axes=(0, 1))  # 去中心化
    imgp = np.real(ifft2(Gp, axes=(0, 1)))  # 反变换 取实部
    imgp = np.uint8(np.clip(imgp, 0, 255))
    imgout = imgp[0:M, 0:N]
    return imgout


def butterworth_show():
    # 巴特沃斯低通滤波器 测试函数
    img_bgr = cv.imread("../Picture/cat1.jpg", cv.IMREAD_COLOR)
    img = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    var = 0.01  # 方差置为0.01
    img_n = noise_img(img, var)
    n = 2  # 初始化阶次
    D0 = 100  # 初始化截止频率
    imgout = butterworth_low(img, n, D0)
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("原图像", fontsize=20)
    plt.axis(False)
    plt.subplot(1, 3, 2)
    plt.imshow(img_n)
    plt.title("方差为{}的高斯噪声图像".format(var), fontsize=20)
    plt.axis(False)
    plt.subplot(1, 3, 3)
    plt.imshow(imgout)
    plt.title("n={0}  D0={1} 低通巴特沃斯滤波图像".format(n, D0), fontsize=20)
    plt.axis(False)
    plt.tight_layout()
    plt.show()


def MotionBlurSim(image, Te, xa, yb):
    # 匀速直线运动图像模糊退化
    rows, cols = image.shape[0:2]
    if image.ndim == 3:  # 彩色图像
        imgex = np.pad(image, ((0, rows), (0, cols), (0, 0)), mode='reflect')
    elif image.ndim == 2:  # 灰度图像
        imgex = np.pad(image, ((0, rows), (0, cols)), mode='reflect')
    img_dft = fftshift(fft2(imgex, axes=(0, 1)), axes=(0, 1))
    # 生成运动模糊退化函数
    # 构建频域平面坐标网格数组
    v = np.arange(-cols, cols)
    u = np.arange(-rows, rows)
    Va, Ua = np.meshgrid(v, u)
    temp = np.pi * (Ua * yb + Va * xa)
    Hmb = Te * np.ones((2 * rows, 2 * cols)).astype(np.complex64)
    indx = np.nonzero(temp)
    Hmb[indx] = np.exp(-1j * temp[indx]) * Te * np.sin(temp[indx]) / temp[indx]
    if image.ndim == 3:
        Hmb = np.dstack((Hmb, Hmb, Hmb))
    # 计算图像DFT与运动模糊退化函数的点积
    Gp = img_dft * Hmb
    Gp = ifftshift(Gp, axes=(0, 1))
    imgp = np.real(ifft2(Gp, axes=(0, 1)))
    imgp = np.uint8(np.clip(imgp, 0, 255))
    imgout = imgp[0:rows, 0:cols]
    return imgout, Hmb


def WinInvFilter(image, H, radius):
    # 加窗逆滤波
    # 当radius取极大时，相当于直接逆滤波
    rows, cols = image.shape[0:2]
    if image.ndim == 3:
        imgex = np.pad(image, ((0, rows), (0, cols), (0, 0)), mode='reflect')
    elif image.ndim == 2:
        imgex = np.pad(image, ((0, rows), (0, cols)), mode='reflect')
    img_dft = fftshift(fft2(imgex, axes=(0, 1)), axes=(0, 1))
    # 加窗逆滤波
    Gp = img_dft.copy()
    # 构建频域平面坐标网格数组
    v = np.arange(-cols, cols)
    u = np.arange(-rows, rows)
    Va, Ua = np.meshgrid(v, u)
    D2 = Ua ** 2 + Va ** 2
    # 计算圆形窗区域内的元素索引
    indx = D2 <= radius ** 2
    Gp[indx] = img_dft[indx] / (H[indx] + np.finfo(np.float32).eps)
    Gp = ifftshift(Gp, axes=(0, 1))
    imgp = np.real(ifft2(Gp, axes=(0, 1)))
    imgp = np.uint8(np.clip(imgp, 0, 255))
    imgout = imgp[0:rows, 0:cols]
    return imgout


def wiener_filter(img, H, NSR):
    # 维纳逆滤波
    rows, cols = img.shape[0:2]
    if img.ndim == 3:
        imgex = np.pad(img, ((0, rows), (0, cols), (0, 0)), mode='reflect')
    elif img.ndim == 2:
        imgex = np.pad(img, ((0, rows), (0, cols)), mode='reflect')
    img_dft = fftshift(fft2(imgex, axes=(0, 1)), axes=(0, 1))
    # 计算维纳滤波复原图像的频谱
    Gp = img_dft * np.conj(H) / (np.abs(H) ** 2 + NSR + np.finfo(np.float32).eps)
    Gp = ifftshift(Gp, axes=(0, 1))
    imgp = np.real(ifft2(Gp, axes=(0, 1)))
    imgp = np.uint8(np.clip(imgp, 0, 255))
    img_res = imgp[0:rows, 0:cols]
    return img_res


def restore_show():
    # 图像复原处理测试函数
    img_bgr = cv.imread("../Picture/cat1.jpg", cv.IMREAD_COLOR)
    img = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    Te = 1  # 曝光时间
    xa = 0.02  # 运动速度
    yb = 0.02
    img_M, Hmb = MotionBlurSim(img, Te, xa, yb)
    var = 0.001  # 方差为0.001
    img_M_G = noise_img(img_M, var)
    radius = 5  # 截止频率
    img_res1 = WinInvFilter(img_M_G, Hmb, radius)
    NSR = 0.005  # 图像信噪比
    img_res2 = wiener_filter(img_M_G, Hmb, NSR)
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title("原图像", fontsize=20)
    plt.axis(False)
    plt.subplot(2, 2, 2)
    plt.imshow(img_M_G)
    plt.title("退化图像", fontsize=20)
    plt.axis(False)
    plt.subplot(2, 2, 3)
    plt.imshow(img_res1)
    plt.title("radius={} 加窗逆滤波复原图像".format(radius), fontsize=20)
    plt.axis(False)
    plt.subplot(2, 2, 4)
    plt.imshow(img_res2)
    plt.title("NSR={} 维纳滤波复原图像".format(NSR), fontsize=20)
    plt.axis(False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    fourier_show()
    butterworth_show()
    restore_show()
