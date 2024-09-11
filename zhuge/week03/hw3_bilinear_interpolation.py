import numpy as np
import cv2 

def bilinear_interpolation(img, out_dim):
    src_h,src_w,channel = img.shape[:3]
    print(f"h is {src_h}, w is {src_w}, channel is {channel}")
    dst_h, dst_w = out_dim[1], out_dim[0]
    dst_img = np.zeros((dst_h,dst_w,channel),dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(channel):
        for dst_x in range(dst_w):
            for dst_y in range(dst_h):
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                src_x0, src_y0 = int(src_x), int(src_y)
                src_x1, src_y1 = min(src_x0+1, src_w-1), min(src_y0+1, src_h-1)
                
                temp0 = (src_x1 - src_x) * img[src_y0,src_x0,i] + (src_x - src_x0) * img[src_y0,src_x1,i]
                temp1 = (src_x1 - src_x) * img[src_y1,src_x0,i] + (src_x - src_x0) * img[src_y1,src_x1,i]
                dst_img[dst_y,dst_x,i] = (src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1
    return dst_img


img = cv2.imread("/Users/yuyaozhuge/Documents/AI学习/【2】数学&数字图像/lenna.png")
out_dim = (700,700)
cv2.imshow("bilinear interpolation show", bilinear_interpolation(img,out_dim))
cv2.waitKey(0)

