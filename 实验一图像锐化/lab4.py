import  cv2
from pylab import *
import numpy as np


def Laplace(Image,a):                   #输入图像Image，模板a
    im=array(Image)
    img=array(Image)
    dim=math.sqrt(len(a))                   # 模板的维度dim
    w=im.shape[0]                           # 计算输入图像的宽高
    h=im.shape[1]
    b=[]                                    # 待处理的与模板等大小的图像块,分BGR通道
    g=[]
    r=[]
    if sum(a)==0:                           # 判断模板a的和是否为0
        A=1
    else:
        A=sum(a)
        for i in range(int(dim/2),w-int(dim/2)):
            for j in range(int(dim/2),h-int(dim/2)):
                for m in range(-int(dim/2),-int(dim/2)+int(dim)):
                    for n in range(-int(dim / 2), -int(dim / 2) + int(dim)):
                        b.append(im[i+m,j+n,0])
                        g.append(im[i+m,j+n,1])
                        r.append(im[i+m,j+n,2])
                img[i,j,0]=sum(np.multiply(np.array(a),np.array(b)))/A
                img[i, j, 1] =sum(np.multiply(np.array(a),np.array(g)))/A
                img[i, j, 2] =sum(np.multiply(np.array(a),np.array(r)))/A
                b=[];g=[];r=[]

    return img
img=cv2.imread('4.jpg')
x=1
a=[0,-x,0,-x,1+4*x,-x,0,-x,0]
b=[-x,-x,-x,-x,1+8*x,-x,-x,-x,-x]
im=Laplace(img,b)
cv2.imshow('Origin',img)
cv2.imshow('Laplace',im)
cv2.imwrite("result2.jpg",im)

cv2.waitKey()
cv2.destroyAllWindows()
