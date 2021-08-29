import cv2
import numpy as np
import matplotlib.pyplot as plt


def hist_equal(img, z_max=255):
    H, W = img.shape
    # S is the total of pixels
    S = H * W * 1.
    out = img.copy()
    sum_h = 0
    for i in range(1, 255):
        ind = np.where(img == i)
        sum_h += len(img[ind])
        z_prime = z_max / S * sum_h
        out[ind] = z_prime
    out = out.astype(np.uint8)
    return out

img = cv2.imread('3.jpg', cv2.IMREAD_GRAYSCALE)
equ  =hist_equal(img)
cv2.imshow("src", img)
cv2.imshow("result", equ)
cv2.imwrite("result3.jpg",equ)
plt.hist(img.ravel(), 256)
plt.figure()
plt.hist(equ.ravel(), 256)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()