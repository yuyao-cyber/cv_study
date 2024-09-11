import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("/Users/yuyaozhuge/Documents/AI学习/【2】数学&数字图像/lenna.png")
img_gray = cv2.imread("/Users/yuyaozhuge/Documents/AI学习/【2】数学&数字图像/lenna.png", 0)

# #equalizedHist
# dst = cv2.equalizeHist(img_gray)
# # dst_hist = cv2.calcHist([dst],[0],None,[256],[0,256])
# plt.figure()
# plt.hist(dst.ravel(),256)
# plt.xlim((0,256))
# plt.show()

#equlized Hist manually
print(img_gray)
h,w = img_gray.shape[:2]
hist, bins = np.histogram(img_gray.ravel(), 256, [0,256])
cdf = hist.cumsum()
norm_cdf = cdf*255/cdf[-1]
img_equalized = np.interp(img_gray.flatten(),bins[:-1],norm_cdf).reshape(img_gray.shape)

img_equalized = img_equalized.astype(np.uint8)
cv2.imshow("Orignal VS. Equalized", np.hstack([img_gray, img_equalized]))
cv2.waitKey(0)
cv2.destroyAllWindows()

#彩色图像直方图
(b,g,r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
result = cv2.merge((bH,gH,rH))
cv2.imshow("equalized lenna: ", result)
cv2.waitKey(0)