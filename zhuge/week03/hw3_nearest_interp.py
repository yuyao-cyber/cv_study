import cv2
import numpy as np

def nearest_interp(img,h_new,w_new):
    # print(img)
    h,w,c = img.shape[:3]
    empty_img = np.zeros((h_new, w_new,c),dtype= np.uint8)
    scale_x = h_new / h
    scale_y = w_new / w
    for i in range(h_new):
        for j in range(w_new):
            y = int(j / scale_y + 0.5)
            x = int(i / scale_x + 0.5)
            empty_img[i,j] = img[x,y]
    return empty_img


img = cv2.imread("/Users/yuyaozhuge/Documents/AI学习/【2】数学&数字图像/lenna.png")
h_new, w_new = 600, 800
img_new = nearest_interp(img,h_new,w_new)
cv2.imshow("nearest_interp",img_new)
cv2.waitKey(0)
