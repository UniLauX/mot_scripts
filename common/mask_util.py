import cPickle
import numpy as np
##------------------------------------ mask util --------------------------------------------------
## write binary mask into file_path (.sm format)
def save_binary_mask(file_path, bin_mask):
    with open(file_path, 'wb') as f_save:
        cPickle.dump(bin_mask, f_save, cPickle.HIGHEST_PROTOCOL)

## load  binary mask from file_path
def load_binary_mask(file_path):
    with open(file_path, 'rb') as f:
        bin_mask = cPickle.load(f)
        return bin_mask

## get a maxbox-size mask from org-size mask
def get_maxbox_mask(label_mask,bbox,max_bbox):
    x_min,y_min,x_max,y_max=max_bbox
    w_max=x_max-x_min+1
    h_max=y_max-y_min+1

    ##for current image:
    im_x1=max(0,int(np.round(bbox[0])))
    im_y1=max(0,int(np.round(bbox[1])))
    im_w=label_mask.shape[1]
    im_h=label_mask.shape[0]
    im_x2=im_x1+im_w-1
    im_y2=im_y1+im_h-1

    max_label_mask=np.zeros((h_max, w_max), dtype=np.uint8) 
    max_label_mask[im_y1-y_min:im_y2-y_min+1,im_x1-x_min:im_x2-x_min+1]=label_mask
    return max_label_mask

## get mask which has the same size with bbox from max_mask
def get_minbox_mask(max_mask,bbox,max_bbox=[]):
    if len(max_bbox)==0:
        h_max,w_max=max_mask.shape
        max_bbox=[0,0,w_max-1,h_max-1]

    x_min,y_min,x_max,y_max=max_bbox
    x1_new,y1_new,x2_new,y2_new=bbox
    w_new=x2_new-x1_new+1
    h_new=y2_new-y1_new+1   
    mask=np.zeros((h_new, w_new), dtype=np.uint8) 
    mask=max_mask[y1_new-y_min:y2_new-y_min+1,x1_new-x_min: x2_new-x_min+1]       
    return mask

## tailor masks based on boxes's shape
def masks_align_boxes(in_masks,bboxes,max_bboxes=[]):
    mask_num=len(in_masks)
    out_masks=[]
    for n_id in xrange(mask_num):
        i_mask=in_masks[n_id]
        bbox=bboxes[n_id]
        if len(max_bboxes)==0:           ## large to small
            o_mask=get_minbox_mask(i_mask,bbox)
            out_masks.append(o_mask)
        else:
            max_bbox=max_bboxes[n_id]    ## small to large
            o_mask=get_maxbox_mask(i_mask,bbox,max_bbox)  ## haven't test
            out_masks.append(o_mask)
    return out_masks           

## save masks_tensor(.npy) into masks_path
def save_masks_tensor(masks_path,masks_tensor):
    np.save(masks_path,masks_tensor)

## load masks_tensor(.npy) from masks_path
def load_masks_tensor(masks_path):
    masks_tensor=np.load(gt_masks_path)
    return masks_tensor






