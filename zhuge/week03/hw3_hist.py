import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("/Users/yuyaozhuge/Documents/AI学习/【2】数学&数字图像/lenna.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Method 1
# plt.figure()
# plt.hist(img_gray.ravel(),256)
# plt.show()

#Method 2
# hist = cv2.calcHist([img_gray],[0],None,[256],[0,256])
# plt.figure()
# plt.title("histogram")
# plt.xlabel("bins")
# plt.ylabel("# of pixels")
# plt.plot(hist)
# plt.xlim((0,256))
# plt.show()

#彩色图像直方图
chans = cv2.split(img)
colors = ("b","g","r")

plt.figure()
plt.title("Flattened color histogram")
plt.xlabel("bins")
plt.ylabel("# of pixels")
for (chan,color) in zip(chans,colors):
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist,color = color)
    plt.xlim((0,256))
plt.show()