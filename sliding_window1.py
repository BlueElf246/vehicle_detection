from load_dataset import get_feature_of_image, change_color_space
import cv2
import numpy as np
import pickle
from setting  import win_size
from imutils.object_detection import  non_max_suppression
def load_classifier():
    d= pickle.load(open('lp_detect.p', 'rb'))
    return d
def sliding_window(img,params,scale, y_start_stop=[None, None], cell_per_step=2):
    bbox=[]
    if y_start_stop[0] == None or y_start_stop[0] > img.shape[0]:
        y_start_stop[0]=0
    if y_start_stop[1]== None  or y_start_stop[1] > img.shape[0]:
        y_start_stop[1]= img.shape[0]
    win_x, win_y, channel= params['size_of_pic_train']
    pix_per_cell= params['pix_per_cell']
    svc= params['svc']
    scaler= params['scaler']
    shape_img= img.shape
    if scale !=1:
        img=cv2.resize(img,(np.int32(shape_img[1]/scale), np.int32(shape_img[0]/scale)))
        shape_img = img.shape
    number_of_cell_in_x= (shape_img[1]//pix_per_cell)-1
    number_of_cell_in_y= (shape_img[0]//pix_per_cell)-1

    number_of_cell_per_window_x= (win_x//pix_per_cell)-1
    number_of_cell_per_window_y= (win_y//pix_per_cell)-1

    number_of_window_in_x= (number_of_cell_in_x-number_of_cell_per_window_x)//cell_per_step
    number_of_window_in_y= (number_of_cell_in_y-number_of_cell_per_window_y)//cell_per_step


    h= get_feature_of_image(img, orient=params['orient'], pix_per_cell=params['pix_per_cell'], cell_per_block=params['cell_per_block'],hog_fea=params['hog_feat'],
                                     spatial_size=params['spatial_size'], spatial_fea=params['spatial_feat'],bins=params['hist_bins'], color_fea=params['hist_feat'],
                                feature_vector=False, special=True)
    ch1=h[0]
    ch2=h[1]
    ch3=h[2]

    for y in range(number_of_window_in_y+1):
        for x in range(number_of_window_in_x+1):
            x_pos= x*cell_per_step
            y_pos= y*cell_per_step
            hog_fea1= ch1[y_pos:y_pos+number_of_cell_per_window_y, x_pos:x_pos+number_of_cell_per_window_x].ravel()
            hog_fea2= ch2[y_pos:y_pos+number_of_cell_per_window_y, x_pos:x_pos+number_of_cell_per_window_x].ravel()
            hog_fea3= ch3[y_pos:y_pos + number_of_cell_per_window_y, x_pos:x_pos + number_of_cell_per_window_x].ravel()
            hog_f= np.hstack((hog_fea1,hog_fea2,hog_fea3))
            x_top= x_pos * pix_per_cell
            y_top= y_pos * pix_per_cell


            img_crop= cv2.resize(img[y_top:y_top+win_y,x_top:x_top+win_x],(win_x,win_y))
            spatial_feature = get_feature_of_image(img_crop, hog_fea=False, spatial_fea=params['spatial_feat'], spatial_size=params['spatial_size'], color_fea=False)
            color_feature   = get_feature_of_image(img_crop, hog_fea=False, spatial_fea=False, bins=params['hist_bins'], color_fea=params['hist_feat'])
            feature= np.concatenate((hog_f, color_feature, spatial_feature))
            scaled_feature=scaler.transform(np.array(feature).reshape(1,-1))
            prediction= svc.predict(scaled_feature)
            if prediction ==1:
                x_start=np.int32(x_top*scale)
                y_start=np.int32(y_top*scale+y_start_stop[0])
                x_end  =np.int32((x_top+win_x)*scale)
                y_end  =np.int32(scale*(y_top+win_y)+y_start_stop[0])
                bbox.append([x_start,y_start,x_end, y_end])
    return bbox
def find_car_multi_scale(img,params, win_size):
    bboxes=[]
    print('number of scale use:', len(win_size['use_scale']))
    win_scale=win_size['use_scale']
    if 0 in win_scale:
        print('yes')
        y_start=win_size['scale_0'][0]
        y_stop=win_size['scale_0'][1]
        scale_0=win_size['scale_0'][2]
        bboxes.append(sliding_window(img, params=params, y_start_stop=[None,None], cell_per_step=2, scale=0.8))
    if 1 in win_scale:
        y_start = win_size['scale_1'][0]
        y_stop = win_size['scale_1'][1]
        scale_0 = win_size['scale_1'][2]
        bboxes.append(
            sliding_window(img, params=params, y_start_stop=[y_start, y_stop], cell_per_step=2, scale=scale_0))
    if 2 in win_scale:
        y_start = win_size['scale_2'][0]
        y_stop = win_size['scale_2'][1]
        scale_0 = win_size['scale_2'][2]
        bboxes.append(
            sliding_window(img, params=params, y_start_stop=[y_start, y_stop], cell_per_step=2, scale=scale_0))
    if 3 in win_scale:
        y_start = win_size['scale_3'][0]
        y_stop = win_size['scale_3'][1]
        scale_0 = win_size['scale_3'][2]
        bboxes.append(
            sliding_window(img, params=params, y_start_stop=[y_start, y_stop], cell_per_step=2, scale=scale_0))
    bboxes= np.concatenate(bboxes)
    print(len(bboxes))
    return bboxes,non_max_suppression(np.array(bboxes),probs=None, overlapThresh=0.2)
def draw(img,box):
    for x in box:
        cv2.rectangle(img, (x[0],x[1]), (x[2],x[3]), (0,0,255), 2)
    return img
params=load_classifier()
img   = cv2.imread('test1.jpg')
img1  = img.copy()
img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
bbox, bbox_nms  = find_car_multi_scale(img,params, win_size)

img   =draw(img, bbox)
img1  =draw(img1, bbox_nms)
i= np.concatenate((img,img1),axis=0)
cv2.imshow('i',i)
cv2.waitKey(0)