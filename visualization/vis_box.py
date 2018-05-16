## This function is for visualize different boxes' cases
from mot_config import mcfg
import numpy as np
import cv2
import os

im_ext=mcfg.DATA.IMGEXT='.jpg'
mid_line_width=2
## vis match boxes to check if it's correct
def vis_match_boxes(vis_dir,im_names,ims,boxes,shift_vec,color1=[0,0,255],color2=[255,0,0]):
    print '============================= vis match boxes================================'
    im_num=len(im_names)
    shift_boxes=np.zeros((im_num-1,4),dtype=int)
    
    for n_id in xrange(im_num):
        im_name=im_names[n_id]
        im=ims[n_id]
        box=boxes[n_id]
        im_path=os.path.join(vis_dir,im_name+im_ext)
        vis_im=im.copy()
        if n_id==0:
            cv2.rectangle(vis_im,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),color1,mid_line_width) 
            cv2.imwrite(im_path,vis_im)
            #  cv2.imshow('vis_im',vis_im)
            #  cv2.waitKey(-1)
        else:
            center_shift=np.zeros(4,)
            center_shift[:2]=center_shift[2:]=shift_vec[n_id-1]
            s_box=boxes[n_id-1]+center_shift
            
            cv2.rectangle(vis_im,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),color1,mid_line_width) 
            cv2.rectangle(vis_im,(int(s_box[0]),int(s_box[1])),(int(s_box[2]),int(s_box[3])),color2,mid_line_width) 
            cv2.imwrite(im_path,vis_im)



