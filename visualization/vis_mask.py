import cv2
import numpy as np
from vis_util import color_map
## parameters
thresh_mask=0.5
alpha=0.3

## vis one mask in one image
## mask has the same size with img
## mask is binary mask  
def vis_im_mask(im,mask,color=[0,0,255],return_flag=False):
    mask_t = np.zeros((im.shape), dtype=np.uint8)
    id_fg=(mask >= thresh_mask)

    mask_t[id_fg, :] =color
    cv2.addWeighted(mask_t, alpha,im, 1 - alpha, 0, mask_t)

    if return_flag:
        return mask_t
    else:
        cv2.imshow('vis_mask',mask_t)
        cv2.waitKey(-1)

## vis binary mask with (black-white look)
def vis_alpha_mask(mask,return_flag=False):
    vis_mask=mask*255.0
    if return_flag:
        return vis_mask
    else:
        cv2.imshow('vis_mask',vis_mask)
        cv2.waitKey(-1)

## copy from track_mask.py, need to be concised
def vis_final_link_mask(init_mask,final_mask,im,max_bbox,im_path):
    print 'vis final link mask...'
    print 'im_path:', im_path
     ##picked based on CRF
    color_mBox=[128,128,128] 
    color_init_mask=[255,0,0]
    color_pk_mask=[0,0,255]
    font = cv2.FONT_HERSHEY_SIMPLEX

    thresh_mask=0.5
    alpha=0.3
    im_width=im.shape[1]
    im_height=im.shape[0]

    #for max_box(through the whole nearby 20f frames)
    x_min=max_bbox[0]
    y_min=max_bbox[1]
    x_max=max_bbox[2]
    y_max=max_bbox[3]
    w_max=x_max-x_min+1
    h_max=y_max-y_min+1

    ##(1) mask from Mask-RCNN
    max_label_mask=mask1
    im_vis=im.copy()
    label_mask_t = np.zeros((h_max, w_max, 3), dtype=np.uint8)
    label_id_fg=(max_label_mask >= thresh_mask)
    label_im_box=im_vis[y_min:y_max + 1, x_min:x_max + 1, :] #rect im_box
    label_mask_t[:,:,:]=im_vis[y_min:y_max + 1, x_min:x_max + 1, :]      
    label_mask_t[label_id_fg, :] =color_init_mask
    cv2.addWeighted(label_mask_t, alpha, label_im_box, 1 - alpha, 0, label_im_box)
    cv2.rectangle(label_im_box,(0,0),(w_max,h_max),color_mBox,2) 
    cv2.putText(label_im_box, 'maskrcnn mask',(int(w_max/4),int(12)),font,0.5,(255,255,255),2)  

    ##(2) maxflow picked mask(unary and pairwise)
    picked_mask=final_mask
    pk_im_vis=im.copy()
    pk_label_mask_t = np.zeros((h_max, w_max, 3), dtype=np.uint8)

    pk_label_id_fg=(picked_mask >= thresh_mask)
    pk_label_im_box=pk_im_vis[y_min:y_max + 1, x_min:x_max + 1, :] #rect im_box
    pk_label_mask_t[:,:,:]=pk_im_vis[y_min:y_max + 1, x_min:x_max + 1, :]      
    pk_label_mask_t[pk_label_id_fg, :] =color_pk_mask
    cv2.addWeighted(pk_label_mask_t, alpha, pk_label_im_box, 1 - alpha, 0, pk_label_im_box)
    ##cv2.rectangle(label_im_box,(im_x1-x_min,im_y1-y_min),(im_x2-x_min,im_y2-y_min),color_box,1)   # predicted proposal
    cv2.rectangle(pk_label_im_box,(0,0),(w_max,h_max),color_mBox,2)   #max_box 
    cv2.putText(pk_label_im_box, 'refine mask',(int(w_max/4),int(12)),font,0.5,(255,255,255),2)  

     ## link image
    link_im= np.zeros((h_max, w_max*2, 3), dtype=np.uint8)
    link_im[:,0:w_max,:]=label_im_box
    link_im[:,w_max:w_max*2,:]=pk_label_im_box
    cv2.imwrite(im_path,link_im)