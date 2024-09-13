import numpy as np
import cv2
from numpy import shape
import random

def gaussion(img, mean, sigma, percentage):
    img_noise = img
    h, w = img.shape[0], img.shape[1]
    num = int(h * w * percentage)
    all_pixels = [(i,j) for i in range(h) for j in range(w)]
    random.shuffle(all_pixels)
    shuffled_pixels = all_pixels[:num]
    for (randX,randY) in shuffled_pixels:
        img_noise[randX, randY] = img_noise[randX, randY] + random.gauss(mean, sigma)
        if img_noise[randX, randY] > 255:
            img_noise[randX, randY] = 255
        elif img_noise[randX, randY] < 0:
            img_noise[randX, randY] = 0
    return img_noise

img = cv2.imread("/Users/yuyaozhuge/Documents/AI学习/【2】数学&数字图像/lenna.png",0)
img_noise = gaussion(img, 2, sigma=4, percentage=0.9)
img = cv2.imread("/Users/yuyaozhuge/Documents/AI学习/【2】数学&数字图像/lenna.png",0)
# img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original VS Gaussion Noise", np.hstack([img,img_noise]))
cv2.waitKey(0)