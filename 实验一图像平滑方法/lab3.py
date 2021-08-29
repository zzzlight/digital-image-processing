import cv2
from  pylab import  *

#一次不要调用太多 否则中值滤波和均值滤波都运行跑起来速度会很慢才出来
#中值滤波
def median_filter(image, win):
    H, W, C = image.shape
    result = image.copy()
    for h in range(1, H-2):
        for w in range(1, W-2):
            for c in range(C):
                result[h, w, c] = np.median(result[h:h+win, w:w+win, c])
    return result

img=cv2.imread("noise2.jpg")

resultMedian1=median_filter(img,5)
resultMedian2=median_filter(img,7)
cv2.imshow("orgin",img)
cv2.imshow("resultMedian5",resultMedian1)
cv2.imshow("resultMedian7",resultMedian2)
cv2.imwrite("resultMedian1.jpg",resultMedian1)
cv2.imwrite("resultMedian2.jpg",resultMedian2)



"""
#均值滤波
def meansBlur(src, ksize):
 dst = np.copy(src) #创建输出图像
 kernel = np.ones((ksize, ksize)) # 卷积核
 padding_num = int((ksize - 1) / 2) #需要补0
 dst = np.pad(dst, (padding_num, padding_num), mode="constant", constant_values=0)
 w, h,C = dst.shape
 dst = np.copy(dst)
 for i in range(padding_num, w - padding_num):
  for j in range(padding_num, h - padding_num):
      for c in range(C):
        dst[i, j,c] = np.sum(kernel * dst[i - padding_num:i + padding_num + 1, j - padding_num:j + padding_num + 1, c]) \
        // (ksize ** 2)
 dst = dst[padding_num:w - padding_num, padding_num:h - padding_num] #把操作完多余的0去除，保证尺寸一样大
 return dst

def mean_filter(image,ksize):
    K = np.ones((ksize, ksize))
    K = np.array(K)
    H, W, C = image.shape
    result = image.copy()
    # 因为卷积核是以左上角为定位，所以遍历时最后要停到H-（ksize-1）处
    for h in range(1, H-(ksize-1)):   #5时候参数为h-4  3时候为h-2 7为h-6
        for w in range(1, W-(ksize-1)):
            for c in range(C):
                result[h, w, c] = sum(sum(K * result[h:h+K.shape[0], w:w+K.shape[1], c])) // (ksize ** 2)
    #边框没处理
    return result

resultMean5=mean_filter(img,5)
resultMean7=mean_filter(img,7)
cv2.imshow("resultMean5",resultMean5)
cv2.imshow("resultMean7",resultMean7)
cv2.imwrite("resultMean1.jpg",resultMean5)
cv2.imwrite("resultMean2.jpg",resultMean7)
"""





cv2.waitKey()
cv2.destroyAllWindows()