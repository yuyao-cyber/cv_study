import matplotlib.font_manager as fm
import cv2
import matplotlib.pyplot as plt
import numpy as np

font_path = '/System/Library/Fonts/Supplemental/Songti.ttc'  # Update this to the path of your font file
prop = fm.FontProperties(fname=font_path)
img = cv2.imread('lenna.png', 0)
print(img.shape)
h, w = img.shape[:]
data = img.reshape((h* w))
data = np.float32(data)
K = 4
bestLabels = None
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1)
attempts = 10
flags = cv2.KMEANS_RANDOM_CENTERS

compactness, labels, centers = cv2.kmeans(data, K, bestLabels, criteria, attempts, flags)

dst = labels.reshape((h,w))
titles = [u'原始图像',u'聚类图像']
images = [img,dst]
for i in range(2):  
    plt.subplot(1,2,i+1),
    plt.imshow(images[i],'gray')
    plt.title(titles[i],fontproperties=prop)
plt.show()
