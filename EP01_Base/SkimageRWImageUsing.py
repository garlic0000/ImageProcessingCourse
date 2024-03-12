from skimage import io, color, util, data
import matplotlib.pyplot as plt


def skimagerw():
    img = io.imread("../Picture/xiangrikui.jpg")
    img_gray = color.rgb2gray(img)
    img_gray = util.img_as_ubyte(img_gray)
    io.imsave("../Picture/gray_xiangrikui.jpg", img_gray)
    io.imshow(img)
    # 没有io.show() 图片不会显示
    io.show()
    io.imshow(img_gray)
    io.show()


def skshow():
    # 这些图片需要进行转换
    img1 = data.camera()
    img1 = util.img_as_ubyte(img1)
    img2 = data.lily()
    img2 = util.img_as_ubyte(img2)
    img3 = data.coins()
    img3 = util.img_as_ubyte(img3)
    img4 = data.coffee()
    img4 = util.img_as_ubyte(img4)

    imgs = [img1, img2, img3, img4]
    fig, axe = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    pos = 0
    for i in range(2):
        for j in range(2):
            # 设置单通道图像的显示方式为灰度图像 不影响RGB彩色图像的显示
            axe[i, j].imshow(imgs[pos], cmap='gray', vmin=0, vmax=255)
            axe[i, j].axis(False)
            pos = pos + 1
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    skshow()

