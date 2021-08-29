import cv2
import numpy as np
import matplotlib.pyplot as plt

def dctpad(src):
    h, w = src.shape[:2]
    h1 = int((h / 8 + 0.5)) * 8
    w1 = int((w / 8 + 0.5)) * 8
    # 拷贝到补充的数组
    dft_A = np.zeros((h1, w1), dtype=np.float32)
    dft_A[:h, :w] = src
    return dft_A

image = ["1.jpg","2.jpg","3.jpg","4.jpg"]
stage = ["Input Image" ,"output Image"]
plt.figure(figsize=(15 ,15))
for i, element in enumerate(image):
    img2 = cv2.imread(element ,0)

    # 图片的高度和宽度
    h, w = img2.shape[:2]

    imgf = dctpad(img2)
    dct = np.zeros_like(imgf)
    imsize = imgf.shape
    # 8x8 DCT
    for x in range(0, imgf.shape[0], 8):
        for y in range(0, imgf.shape[1], 8):
            dct[x:(x + 8), y:(y + 8)] = cv2.dct(imgf[x:(x + 8), y:(y + 8)])
    # 阈值分割，相当于量化
    thresh = 0.05
    dct_thresh = dct * (abs(dct) > (thresh * np.max(dct)))

    im_dct = np.zeros(imsize)
    # 8x8 IDCT
    for x in range(0, im_dct.shape[0], 8):
        for y in range(0, im_dct.shape[1], 8):
            im_dct[x:(x + 8), y:(y + 8)] = cv2.idct(dct_thresh[x:(x + 8), y:(y + 8)])

    images = [img2 ,im_dct]
    percent_nonzeros = [np.sum(dct_thresh != 0.0) / (imsize[0] * imsize[1] * 1.0)]
    for j, img in  enumerate(images):
        plt.subplot(len(image), len(stage), 1 + i * len(stage) + j)
        plt.imshow(img ,cmap = 'gray')
        plt.title(stage[j] ,color='blue')
plt.show()