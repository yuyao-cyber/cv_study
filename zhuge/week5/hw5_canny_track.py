# import numpy as np
# import cv2

# def canny_track(low_threshold):
#     detected_edge = cv2.GaussianBlur(gray,(3,3),0)
#     detected_edge = cv2.Canny(detected_edge,
#                               low_threshold,
#                               low_threshold*ratio,
#                               apertureSize = kernel_size)
#     dst = cv2.bitwise_and(img,img,mask=detected_edge)
#     cv2.imshow("canny demo", dst)

# img = cv2.imread("lenna.png")
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# low_threshold = 0
# max_threshold = 100
# ratio = 3
# kernel_size = 3

# cv2.namedWindow('canny demo')

# cv2.createTrackbar('Min threshold', 'canny demo', low_threshold, max_threshold, canny_track)
# canny_track(low_threshold)
# cv2.waitKey(0)

import cv2
import numpy as np

def CannyThreshold(low_threshold):
    detected_edges = cv2.GaussianBlur(gray,(3,3),0)
    detected_edges = cv2.Canny(detected_edges,
                               low_threshold,
                               low_threshold*ratio,
                               apertureSize = kernel_size)
    dst = cv2.bitwise_and(img, img, mask=detected_edges)
    cv2.imshow('canny show', dst)

low_threshold = 0
ratio = 3
kernel_size = 3
img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
max_LowThreshold = 200

cv2.namedWindow('canny show')
cv2.createTrackbar('Min', 'canny show', low_threshold, max_LowThreshold, CannyThreshold)
CannyThreshold(0)
cv2.waitKey(0)

