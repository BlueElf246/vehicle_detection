from sliding_window1 import *
import time
params=load_classifier()
img   = mpimg.imread('test1.jpg')
img1  = img.copy()
start= time.time()
bbox, bbox_nms  = find_car_multi_scale(img,params, win_size)
end= time.time()
print(f'time is: {end-start}')
img   =draw(img, bbox)
img1  =draw(img1, bbox_nms)
i= np.concatenate((img,img1),axis=0)
cv2.imshow('i',i)
cv2.waitKey(0)