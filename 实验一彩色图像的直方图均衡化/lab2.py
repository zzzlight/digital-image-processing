import cv2
import numpy as np
import matplotlib.pyplot as plt

def hist_equal(img, z_max=255):
    H, W = img.shape
    S = H * W * 1.
    out = img.copy()
    sum_h = 0.
    for i in range(1, 255):
        ind = np.where(img == i)
        sum_h += len(img[ind])
        z_prime = z_max / S * sum_h
        out[ind] = z_prime
    out = out.astype(np.uint8)
    return out


rgb_img = cv2.imread('4.jpg', cv2.IMREAD_COLOR)
# 进行颜色空间转换
hsi_img2 = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
myI=hsi_img2[:,:,2]
H=hsi_img2[:,:,0]
S=hsi_img2[:,:,1]
temp=hist_equal(myI)
#自己编写函数的均衡化
plt.hist(myI.ravel(), 256)
plt.show()
plt.hist(temp.ravel(), 256)
plt.show()
ans=cv2.merge([H,S,temp])
res=cv2.cvtColor(ans,cv2.COLOR_HSV2BGR)
cv2.imshow("ans",res)
cv2.imwrite("result4.jpg",res)
cv2.waitKey()
cv2.destroyAllWindows()

