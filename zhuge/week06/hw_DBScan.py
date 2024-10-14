import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN

iris = datasets.load_iris()
X = iris.data[:, :4]
print(X.shape)

dbscan = DBSCAN(eps=0.6,min_samples=7)
dbscan.fit(X)
label_pred = dbscan.labels_
print(label_pred)
x0 = X[label_pred==0]
x1 = X[label_pred==1]
x2 = X[label_pred==2]
x3 = X[label_pred==3]

plt.scatter(x0[:,0],x0[:,1],c='red',marker='o',label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')  
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2') 
plt.scatter(x3[:, 0], x3[:, 1], c="yellow", marker='+', label='label3') 

# plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='label0')  

plt.xlabel('sepal length')  
plt.ylabel('sepal width')  
plt.legend(loc=2)  
plt.show()  