import numpy as np
from mask_util import get_maxbox_mask
from box_util import get_mini_box_from_mask

##==================================================================================================##
## forward_warp from first image
## using forward_warp(utilize forward_flow) to propogate prev_mask to get next_mask 
## may cause some holes after mapping
def fw_warp_prop_mask(prev_label_mask,prev_fw_flow,prev_bbox):
    print 'track mask using forward optical flow...'
    ## max box    
    max_bbox=[0,0,prev_fw_flow.shape[1]-1,prev_fw_flow.shape[0]-1]
    x_min,y_min,x_max,y_max=max_bbox
    w_max=x_max-x_min+1
    h_max=y_max-y_min+1
    
    if(prev_label_mask.shape==prev_fw_flow.shape[:2]): ## mask has the same size with image
        print 'mask has the same shape with img...'
        prev_max_label_mask=prev_label_mask
    else:                                              # mask has the same size with bbox
        print 'copy the rect mask to img_size mask...'
        im_x1,im_y1,im_x2,im_y2=prev_bbox
        prev_max_label_mask=get_maxbox_mask(prev_label_mask,prev_bbox,max_bbox)  ## tmp

    # cur_max_track_label_mask (tracking from flow)
    max_track_label_mask=np.zeros((h_max, w_max), dtype=np.uint8)  
    for ay in xrange(h_max):
        for ax in xrange(w_max):
            if prev_max_label_mask[ay,ax]==1:
                u=prev_fw_flow[y_min+ay,x_min+ax,0]
                v=prev_fw_flow[y_min+ay,x_min+ax,1]
                bx=min(w_max-1,max(0,ax+u))
                by=min(h_max-1,max(0,ay+v))
                bx=int(np.round(bx))   ##better to use interplation(later)
                by=int(np.round(by))   ##will cause accumulate errors
                max_track_label_mask[by,bx]=1
   
    ## tracked_new bbox(coords)
    new_bbox=get_mini_box_from_mask(max_track_label_mask)
    x1_new=x_min+new_bbox[0]
    y1_new=y_min+new_bbox[1]
    x2_new=x_min+new_bbox[2]
    y2_new=y_min+new_bbox[3]
    w_new=x2_new-x1_new+1
    h_new=y2_new-y1_new+1
    new_bbox=[x1_new,y1_new,x2_new,y2_new]

    ##track_label_mask
    track_label_mask=np.zeros((h_new, w_new), dtype=np.uint8) 
    track_label_mask=max_track_label_mask[y1_new-y_min:y2_new-y_min+1,x1_new-x_min: x2_new-x_min+1]
    return new_bbox,track_label_mask

## backward_warp from first image
## using backward_warp(utilize backward_flow) to propogate prev_mask to get next_mask 
def bw_warp_prop_mask(prev_label_mask,bw_flow,prev_bbox):
    print 'backward_warp track mask using backward optical flow...'
    ## max(img) box
    max_bbox=[0,0,bw_flow.shape[1]-1,bw_flow.shape[0]-1]
    x_min,y_min,x_max,y_max=max_bbox
    w_max=x_max-x_min+1
    h_max=y_max-y_min+1

    if(prev_label_mask.shape==bw_flow.shape[:2]):
        print 'mask has the same shape with img...'
        prev_max_label_mask=prev_label_mask
    else:
        print 'copy the rect mask to img_size mask...'
        im_x1,im_y1,im_x2,im_y2=prev_bbox
        prev_max_label_mask=get_maxbox_mask(prev_label_mask,prev_bbox,max_bbox)  ## tmp
        print 'prev_bbox:',prev_bbox
        print 'prev_label_mask.shape:',prev_label_mask.shape
    ##max_track_label_mask=np.zeros((h_max, w_max), dtype=np.uint8)
    max_track_label_mask=np.ones((h_max, w_max), dtype=np.uint8) 
    print 'prev_max_label_mask.shape:', prev_max_label_mask.shape
    print 'max_track_label_mask.shape:', max_track_label_mask.shape
    
    for by in xrange(h_max):
        for bx in xrange(w_max):
            u=bw_flow[y_min+by,x_min+bx,0]
            v=bw_flow[y_min+by,x_min+bx,1]

            ## coord in first img 
            ax=min(w_max-1,max(0,bx+u))
            ay=min(h_max-1,max(0,by+v))
            
            ax1=int(np.floor(ax))
            ax2=int(np.ceil(ax))
            x_alpha=ax-ax1

            ay1=int(np.floor(ay))
            ay2=int(np.ceil(ay))
            y_alpha=ay-ay1

            ## get values of four corners
            am11=prev_max_label_mask[ay1,ax1]
            am21=prev_max_label_mask[ay1,ax2]
            
            am12=prev_max_label_mask[ay2,ax1]
            am22=prev_max_label_mask[ay2,ax2]

            ## bilinear interplation
            ix1=am11*(1-x_alpha)+am12*x_alpha
            ix2=am12*(1-x_alpha)+am22*x_alpha

            f_val=ix1*(1-y_alpha)+ix2*y_alpha
            f_val=int(np.round(f_val)) 
            max_track_label_mask[by,bx]=f_val

    ## tracked_new bbox(coords)
    new_bbox=get_mini_box_from_mask(max_track_label_mask)
    x1_new=x_min+new_bbox[0]
    y1_new=y_min+new_bbox[1]
    x2_new=x_min+new_bbox[2]
    y2_new=y_min+new_bbox[3]
    w_new=x2_new-x1_new+1
    h_new=y2_new-y1_new+1
    new_bbox=[x1_new,y1_new,x2_new,y2_new]

    ##track_label_mask
    track_label_mask=np.zeros((h_new, w_new), dtype=np.uint8) 
    track_label_mask=max_track_label_mask[y1_new-y_min:y2_new-y_min+1,x1_new-x_min: x2_new-x_min+1]        
    
    return new_bbox,track_label_mask



