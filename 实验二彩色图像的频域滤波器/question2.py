import cv2
import numpy as np
import matplotlib.pyplot as plt


#选择的高斯高通滤波器
def GaussianLowFilter(image,d):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    def make_transform_matrix(d):
        transfor_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x:(x-1)/2,image.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                def cal_distance(pa,pb):
                    from math import sqrt
                    dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)
                    return dis
                dis = cal_distance(center_point,(i,j))
                #修改函数
                transfor_matrix[i,j] = 1-np.exp(-(dis**2)/(2*(d**2)))
        return transfor_matrix
    d_matrix = make_transform_matrix(d)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
    return new_img




size = [5,20,50,80,250];
image = ["1.jpg","2.jpg","3.jpg","4.jpg"]
plt.figure(figsize=(15,15))

for j,element in enumerate(image):
    img2 = cv2.imread(element,0)
#     img_2 = img2[:,:,[2,1,0]]
    for i, element1 in enumerate(size):
        img_blur = GaussianLowFilter(img2,element1)
        plt.subplot(len(image), 5, 1 + i  + j * 5)
        plt.imshow(img_blur)
        plt.title('size: ' + str(element1))
plt.show()

