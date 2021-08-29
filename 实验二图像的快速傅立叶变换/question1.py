import  cv2
import numpy as np
import matplotlib.pyplot as plt

image = ["1.jpg","2.jpg","3.jpg","4.jpg"]
stage = ["origin", "Magnitude Spectrum"]
plt.figure(figsize=(15, 15))
for i, element in enumerate(image):
    img2 = cv2.imread(element, 0)

    # 快速傅立叶变换
    dft = cv2.dft(np.float32(img2), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    images = [img2, magnitude_spectrum]
    for j, img in enumerate(images):
        plt.subplot(len(image), len(stage), 1 + i * len(stage) + j)
        plt.imshow(img, cmap='gray')
        plt.title(stage[j], color='blue')
plt.show()