# 0201.py
import cv2 
import numpy as np
import matplotlib.pyplot as plt

#1 y_train
nPoints = 40
k = nPoints//2

y_train = np.zeros((nPoints), dtype=np.uint) #label data
y_train[k:] = 1 # y_train[:k] = 0


#2: create x_train: 2D coords  
std1 = (30, 50)
np.random.seed(1)
cov1 = [[std1[0]**2, 0], [0.,  std1[1]**2]]
pts1  = np.random.multivariate_normal(mean=(200, 200), cov=cov1, size=k)

std2 = (30, 40)
cov2 = [[std2[0]**2, 0], [0.,  std2[1]**2]]
pts2  = np.random.multivariate_normal(mean=(300, 300), cov=cov2, size=k)

x_train = np.concatenate([pts1, pts2])

#3: save x_train, y_train with 2 class, 40 points
height, width =  500, 600
np.savez('./data/0201_data40.npz', x_train=x_train, y_train= y_train, size=(height, width))

#4: display using OpenCV
dst = np.full((height, width,3), (255, 255, 255), dtype= np.uint8)
class_color = [(255,0,0), (0,0,255)]
for i in range(nPoints): 
    x, y =  x_train[i, :]
    label = y_train[i]
    cv2.circle(dst,(int(x),int(y)), radius=5, color=class_color[label],thickness=-1)
cv2.imshow('dst', dst)                
#cv2.waitKey()


#5: x_train, y_train with 2 class, 50 points
#5-1: add 10 points 
k = 10  
nPoints += k
label = np.zeros((k), dtype=np.uint32)  # class 0
y_train = np.concatenate([y_train, label], dtype=y_train.dtype) #np.int32

#5-2:  
std = (40, 40)
cov = [[std[0]**2, 0], [0.,  std[1]**2]]
pts  = np.random.multivariate_normal(mean=(350, 150), cov=cov, size=k)
x_train = np.concatenate([x_train, pts], dtype=x_train.dtype) #np.float32
np.savez('./data/0201_data50.npz', x_train=x_train, y_train= y_train, size=(height, width))

#5-3: display data using matplotlib
ax = plt.gca()
ax.set_aspect('equal')
#ax.axis('off')
ax.xaxis.tick_top()
plt.xlim(0, width-1)
plt.ylim(0, height-1)
ax.invert_yaxis() # ax.set_ylim(ax.get_ylim()[::-1]) # the same as OpenCV

ax.scatter(x_train[y_train==0, 0], x_train[y_train==0, 1], 50, 'b', 'o')
ax.scatter(x_train[y_train==1, 0], x_train[y_train==1, 1], 50, 'r', 'o') # 'x'
plt.show()
cv2.destroyAllWindows()
  
