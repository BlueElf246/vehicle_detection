import numpy as np
from load_dataset import get_feature_of_image
import cv2
import pickle
def sliding_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64,64), xy_overlap=(0.5,0.5)):
    if x_start_stop[0]==None:
        x_start_stop[0]=0
    if x_start_stop[1]==None:
        x_start_stop[1]=img.shape[1]
    if y_start_stop[0]==None:
        y_start_stop[0]=0
    if y_start_stop[1]==None:
        y_start_stop[1]=img.shape[0]

    xspan= x_start_stop[1]- x_start_stop[0]
    yspan= y_start_stop[1]- y_start_stop[0]
    # number of x_y pixel per step
    nx_pix_per_step= np.int32(xy_window[0]*(1-xy_overlap[0]))
    ny_pix_per_step = np.int32(xy_window[1] * (1 - xy_overlap[1]))
    # compute the number of window
    nx_window= np.int32(xspan/nx_pix_per_step)-1
    ny_window= np.int32(yspan/ny_pix_per_step)-1

    window_list=[]
    for ys in range(ny_window):
        for xs in range(nx_window):
            x_start= xs*nx_pix_per_step + x_start_stop[0]
            y_start= ys*ny_pix_per_step + y_start_stop[0]
            x_stop = x_start + xy_window[0]
            y_stop = y_start + xy_window[1]

            window_list.append(((x_start,y_start),(x_stop,y_stop)))
    return window_list

def search_window(img, window, params):
    on_window=[]
    clf= params['svc']
    scaler= params['scaler']
    for x in window:
        img_crop= img[x[0][1]:x[1][1], x[0][0]:x[1][0]]
        img_crop= cv2.resize(img_crop,(64,64))
        feature=get_feature_of_image(img_crop, orient=params['orient'], pix_per_cell=params['pix_per_cell'], cell_per_block=params['cell_per_block'],hog_fea=params['hog_feat'],
                                     spatial_size=params['spatial_size'], spatial_fea=params['spatial_feat'],bins=params['hist_bins'], color_fea=params['hist_feat'], feature_vector=True, special=False)
        feature_scaled=scaler.transform(np.array(feature).reshape(1, -1))
        prediction= clf.predict(feature_scaled)
        if prediction ==1:
            on_window.append(x)
    return on_window
def draw(img,box):
    for x in box:
        cv2.rectangle(img, x[0], x[1], (0,0,255), 2)
    return img
def pipeline(img):
    params= pickle.load(open('lp_detect.p', "rb"))
    window= sliding_window(img)
    box=search_window(img, window,params)
    img=draw(img, box)
    cv2.imshow('img',img)
    cv2.waitKey(0)
img= cv2.imread('test1.jpg')
img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pipeline(img)






