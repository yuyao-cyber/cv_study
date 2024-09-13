import numpy as np
import cv2
import random
from numpy import shape

def pepper_noise(img,percentage):
    h,w = img.shape[0], img.shape[1]
    num = int(h*w * percentage)
    all_pixels = [(i,j) for i in range(h) for j in range(w)]
    random.shuffle(all_pixels)
    selected_pixels = all_pixels[:num]
    for (randX, randY) in selected_pixels:
        rand = random.random()
        if rand <= 0.5:
            img[randX,randY] = 0
        else:
            img[randX,randY] = 255
    return img

img = cv2.imread("/Users/yuyaozhuge/Documents/AI学习/【2】数学&数字图像/lenna.png",0)
img_pepper_noise = pepper_noise(img,percentage=0.2)
img = cv2.imread("/Users/yuyaozhuge/Documents/AI学习/【2】数学&数字图像/lenna.png",0)
cv2.imshow("Original VS Pepper Noise", np.hstack([img,img_pepper_noise]))
cv2.waitKey(0)
# cv2.destroyWindows()
