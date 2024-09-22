import numpy as np
import cv2

img = cv2.imread('photo1.jpg')
img_copy = img.copy()
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
m = cv2.getPerspectiveTransform(src, dst)
print('warpMatrix is ', m)

result = cv2.warpPerspective(img_copy, m, (337,488))
cv2.imshow('perspective transformation result', result)
cv2.waitKey(0)
