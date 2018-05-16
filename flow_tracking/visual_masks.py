import numpy as np
import cv2
import os
from mot_config import mcfg
##from track_mask import get_maxbox_mask
## configuration
im_ext=mcfg.DATA.IMGEXT

def get_maxbox_mask(label_mask,bbox,max_bbox):
    #for max_box(through the whole nearby 20f frames)
    x_min=max_bbox[0]
    y_min=max_bbox[1]
    x_max=max_bbox[2]
    y_max=max_bbox[3]
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



def vis_propagated_masks(im,bbox,label_mask,prev_bbox,fw_track_mask,next_bbox,bw_track_mask,vis_im_path,max_box):

    ## vis para
    color_mBox=[128,128,128]
    color_mask=[255,0,0]
    color_box=[255,0,0]

    color_fw_mask=[0,255,0]
    color_fw_box=[0,255,0]

    color_bw_mask=[0,0,255]
    color_bw_box=[0,0,255]

    ##intersection
    color_inter_mask=[0,255,255]
    color_inter_box=[0,255,255]

    thresh_mask=0.5
    alpha=0.3
    im_width=im.shape[1]
    im_height=im.shape[0]

    #for max_box(through the whole nearby 20f frames)
    x_min=max_box[0]
    y_min=max_box[1]
    x_max=max_box[2]
    y_max=max_box[3]
    w_max=x_max-x_min+1
    h_max=y_max-y_min+1

    ##for current image:
    im_x1=max(0,int(np.round(bbox[0])))
    im_y1=max(0,int(np.round(bbox[1])))
    im_w=label_mask.shape[1]
    im_h=label_mask.shape[0]
    im_x2=im_x1+im_w-1
    im_y2=im_y1+im_h-1
   
    ##previous frame
    prev_im_x1=prev_bbox[0]
    prev_im_y1=prev_bbox[1]
    prev_im_x2=prev_bbox[2]
    prev_im_y2=prev_bbox[3]

    
    ##next frame
    next_im_x1=next_bbox[0]
    next_im_y1=next_bbox[1]
    next_im_x2=next_bbox[2]
    next_im_y2=next_bbox[3]
      
    ##noisy mask from mask-rcnn(im)
    im_vis=im.copy() #whole image
    label_mask_t = np.zeros((h_max, w_max, 3), dtype=np.uint8)
    label_im_box=im_vis[y_min:y_max + 1, x_min:x_max + 1, :] #rect im_box
    label_mask_t[:,:,:]=im_vis[y_min:y_max + 1, x_min:x_max + 1, :]   

    max_label_mask=np.zeros((h_max, w_max), dtype=np.uint8) 
    max_label_mask[im_y1-y_min:im_y2-y_min+1,im_x1-x_min:im_x2-x_min+1]=label_mask
    label_id_fg=(max_label_mask >= thresh_mask)
    label_mask_t[label_id_fg, :] =color_mask

    cv2.addWeighted(label_mask_t, alpha, label_im_box, 1 - alpha, 0, label_im_box)
    cv2.rectangle(label_im_box,(im_x1-x_min,im_y1-y_min),(im_x2-x_min,im_y2-y_min),color_box,1)   # predicted proposal
    cv2.rectangle(label_im_box,(0,0),(w_max,h_max),color_mBox,2)   #max_box

  
    ##forward tracked mask
    ft_im_vis=im.copy()
    ft_label_mask_t = np.zeros((h_max, w_max, 3), dtype=np.uint8)
    ft_label_id_fg=(fw_track_mask >= thresh_mask)
    ft_label_im_box=ft_im_vis[y_min:y_max + 1, x_min:x_max + 1, :] 
    ft_label_mask_t[:,:,:]=ft_im_vis[y_min:y_max + 1, x_min:x_max + 1, :]    # copy the proposal region from the color image
    ft_label_mask_t[ft_label_id_fg,:] =color_fw_mask

    cv2.addWeighted(ft_label_mask_t, alpha, ft_label_im_box, 1 - alpha, 0, ft_label_im_box)
    cv2.rectangle(ft_label_im_box,(prev_im_x1-x_min,prev_im_y1-y_min),(prev_im_x2-x_min,prev_im_y2-y_min),color_fw_box,1)   # predicted proposal
    cv2.rectangle(ft_label_im_box,(0,0),(w_max,h_max),color_mBox,2)   #max_box
    
    ft_im_vis[y_min:y_max+1,x_min:x_max+1,:]=ft_label_im_box


    ##backward tracked mask
    bt_im_vis=im.copy()
    bt_label_mask_t = np.zeros((h_max, w_max, 3), dtype=np.uint8)
    bt_label_id_fg=(bw_track_mask >= thresh_mask)
    bt_label_im_box=bt_im_vis[y_min:y_max + 1, x_min:x_max + 1, :] 
    bt_label_mask_t[:,:,:]=bt_im_vis[y_min:y_max + 1, x_min:x_max + 1, :]    # copy the proposal region from the color image
    bt_label_mask_t[bt_label_id_fg,:] =color_bw_mask

    cv2.addWeighted(bt_label_mask_t, alpha, bt_label_im_box, 1 - alpha, 0, bt_label_im_box)
    cv2.rectangle(bt_label_im_box,(next_im_x1-x_min,next_im_y1-y_min),(next_im_x2-x_min,next_im_y2-y_min),color_bw_box,1)   # predicted proposal
    cv2.rectangle(bt_label_im_box,(0,0),(w_max,h_max),color_mBox,2)   #max_box
    bt_im_vis[y_min:y_max+1,x_min:x_max+1,:]=bt_label_im_box
   

    ## intersection mask 
    inter_im_vis=im.copy()

    inter_label_mask_t = np.zeros((h_max, w_max, 3), dtype=np.uint8)

    inter_label_id_fg=((max_label_mask+fw_track_mask+bw_track_mask>=3))

    
    inter_label_im_box=inter_im_vis[y_min:y_max + 1, x_min:x_max + 1, :] 
    inter_label_mask_t[:,:,:]=inter_im_vis[y_min:y_max + 1, x_min:x_max + 1, :] 
    
    inter_label_mask_t[inter_label_id_fg,:] =color_inter_mask
    
    cv2.addWeighted(inter_label_mask_t, alpha, inter_label_im_box, 1 - alpha, 0, inter_label_im_box)
 

    inter_im=inter_label_im_box 

    ## link image
    link_im= np.zeros((h_max, w_max*4, 3), dtype=np.uint8)
    link_im[:,0:w_max,:]=label_im_box
    link_im[:,w_max:w_max*2,:]=ft_label_im_box
    link_im[:,w_max*2:w_max*3,:]=bt_label_im_box    
    link_im[:,w_max*3:w_max*4,:]=inter_label_im_box   

    ##cv2.imshow('link_im',link_im)
    ##cv2.imshow('inter_im',inter_im)
    ##cv2.waitKey(-1)
    return link_im

def vis_link_masks_im(fw_track_mask,prev_bbox, max_label_mask,cur_bbox, bw_track_mask,next_bbox, picked_mask,picked_bbox, picked_label,im,max_bbox):
    link_im=[]

    label=picked_label%3+1 
    
    color_mBox=[128,128,128]
    color_mask=[255,0,0]
    color_box=[255,0,0]

    color_fw_mask=[0,255,0]
    color_fw_box=[0,255,0]

    color_bw_mask=[0,0,255]
    color_bw_box=[0,0,255]

    ##intersection
    color_inter_mask=[0,255,255]
    color_inter_box=[0,255,255]
  
    ##picked based on CRF
    color_pk_mask=[255,255,0]
    color_pk_mask=[255,255,0]

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

    ##(1)forward tracked mask
    ft_im_vis=im.copy()
    ft_label_mask_t = np.zeros((h_max, w_max, 3), dtype=np.uint8)
    im_fw_track_mask=get_maxbox_mask(fw_track_mask,prev_bbox,max_bbox) ## changes
    ft_label_id_fg=(im_fw_track_mask >= thresh_mask)
    ft_label_im_box=ft_im_vis[y_min:y_max + 1, x_min:x_max + 1, :] 
    ft_label_mask_t[:,:,:]=ft_im_vis[y_min:y_max + 1, x_min:x_max + 1, :]    # copy the proposal region from the color image
    ft_label_mask_t[ft_label_id_fg,:] =color_fw_mask
    cv2.addWeighted(ft_label_mask_t, alpha, ft_label_im_box, 1 - alpha, 0, ft_label_im_box)
    ##cv2.rectangle(ft_label_im_box,(prev_im_x1-x_min,prev_im_y1-y_min),(prev_im_x2-x_min,prev_im_y2-y_min),color_fw_box,1)   # predicted proposal
    cv2.rectangle(ft_label_im_box,(prev_bbox[0]-x_min,prev_bbox[1]-y_min),(prev_bbox[2]-x_min,prev_bbox[3]-y_min),color_fw_box,2)   # predicted proposal
    cv2.rectangle(ft_label_im_box,(0,0),(w_max,h_max),color_mBox,2)   #max_box
    ##ft_im_vis[y_min:y_max+1,x_min:x_max+1,:]=ft_label_im_box
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(ft_label_im_box, '1. forward tracked mask',(int(w_max/5),int(12)),font,0.5,(255,255,255),2)  
    

    ##(2)noisy mask from mask-rcnn(with max_bbox size)
    im_vis=im.copy()
    label_mask_t = np.zeros((h_max, w_max, 3), dtype=np.uint8)
    im_max_label_mask=get_maxbox_mask(max_label_mask,cur_bbox,max_bbox)  ## changes
    label_id_fg=(im_max_label_mask >= thresh_mask)
    label_im_box=im_vis[y_min:y_max + 1, x_min:x_max + 1, :] #rect im_box
    label_mask_t[:,:,:]=im_vis[y_min:y_max + 1, x_min:x_max + 1, :]      
    label_mask_t[label_id_fg, :] =color_mask


    cv2.addWeighted(label_mask_t, alpha, label_im_box, 1 - alpha, 0, label_im_box)
    ##cv2.rectangle(label_im_box,(im_x1-x_min,im_y1-y_min),(im_x2-x_min,im_y2-y_min),color_box,1)   # predicted proposal
    cv2.rectangle(label_im_box,(cur_bbox[0]-x_min,cur_bbox[1]-y_min),(cur_bbox[2]-x_min,cur_bbox[3]-y_min),color_mask,2)   # predicted proposal 
    cv2.rectangle(label_im_box,(0,0),(w_max,h_max),color_mBox,2)   #max_box 
    cv2.putText(label_im_box, '2. maskrcnn mask',(int(w_max/4),int(12)),font,0.5,(255,255,255),2)  
    

    ##(3) backward tracked mask
    bt_im_vis=im.copy()
    bt_label_mask_t = np.zeros((h_max, w_max, 3), dtype=np.uint8)
    im_bw_track_mask=get_maxbox_mask(bw_track_mask,next_bbox,max_bbox) ## changes
    bt_label_id_fg=(im_bw_track_mask >= thresh_mask)
    bt_label_im_box=bt_im_vis[y_min:y_max + 1, x_min:x_max + 1, :] 
    bt_label_mask_t[:,:,:]=bt_im_vis[y_min:y_max + 1, x_min:x_max + 1, :]    # copy the proposal region from the color image
    bt_label_mask_t[bt_label_id_fg,:] =color_bw_mask

    cv2.addWeighted(bt_label_mask_t, alpha, bt_label_im_box, 1 - alpha, 0, bt_label_im_box)
    ##cv2.rectangle(bt_label_im_box,(next_im_x1-x_min,next_im_y1-y_min),(next_im_x2-x_min,next_im_y2-y_min),color_bw_box,1)   # predicted proposal
    cv2.rectangle(bt_label_im_box,(next_bbox[0]-x_min,next_bbox[1]-y_min),(next_bbox[2]-x_min,next_bbox[3]-y_min),color_bw_box,2)  
    cv2.rectangle(bt_label_im_box,(0,0),(w_max,h_max),color_mBox,2)   #max_box
    ##bt_im_vis[y_min:y_max+1,x_min:x_max+1,:]=bt_label_im_box
    cv2.putText(bt_label_im_box, '3. backward tracked mask',(int(w_max/5),int(12)),font,0.5,(255,255,255),2)  

    ##(4) crf picked mask(unary and pairwise)
    pk_im_vis=im.copy()
    pk_label_mask_t = np.zeros((h_max, w_max, 3), dtype=np.uint8)

    im_picked_mask=get_maxbox_mask(picked_mask,picked_bbox,max_bbox) ## changes 
    pk_label_id_fg=(im_picked_mask >= thresh_mask)
    pk_label_im_box=pk_im_vis[y_min:y_max + 1, x_min:x_max + 1, :] #rect im_box
    pk_label_mask_t[:,:,:]=pk_im_vis[y_min:y_max + 1, x_min:x_max + 1, :]      
    pk_label_mask_t[pk_label_id_fg, :] =color_pk_mask
    cv2.addWeighted(pk_label_mask_t, alpha, pk_label_im_box, 1 - alpha, 0, pk_label_im_box)
    ##cv2.rectangle(label_im_box,(im_x1-x_min,im_y1-y_min),(im_x2-x_min,im_y2-y_min),color_box,1)   # predicted proposal
    cv2.rectangle(pk_label_im_box,(picked_bbox[0]-x_min,picked_bbox[1]-y_min),(picked_bbox[2]-x_min,picked_bbox[3]-y_min),color_pk_mask,2) 
    cv2.rectangle(pk_label_im_box,(0,0),(w_max,h_max),color_mBox,2)   #max_box 
    
    if label==1:
        labeltext='forward tracked mask'
    if label==2:
        labeltext='maskrcnn mask'
    if label==3:
        labeltext='backward tracked mask'

    cv2.putText(pk_label_im_box, 'picked {:} '.format(label)+labeltext,(int(w_max/6),int(12)),font,0.5,(255,255,255),2)  
    
    ##(5) intersection mask 
    ## intersection mask 
    inter_im_vis=im.copy()
    inter_label_mask_t = np.zeros((h_max, w_max, 3), dtype=np.uint8)
    ##inter_label_id_fg=((max_label_mask+fw_track_mask+bw_track_mask>=3))
    inter_label_id_fg=((im_max_label_mask+im_fw_track_mask+im_bw_track_mask>=3))  ##changes

    inter_label_im_box=inter_im_vis[y_min:y_max + 1, x_min:x_max + 1, :] 
    inter_label_mask_t[:,:,:]=inter_im_vis[y_min:y_max + 1, x_min:x_max + 1, :] 
    inter_label_mask_t[inter_label_id_fg,:] =color_inter_mask
    cv2.addWeighted(inter_label_mask_t, alpha, inter_label_im_box, 1 - alpha, 0, inter_label_im_box)
    ##cv2.rectangle(label_im_box,(im_x1-x_min,im_y1-y_min),(im_x2-x_min,im_y2-y_min),color_box,1)   # predicted proposal
  
    cv2.rectangle(inter_label_im_box,(cur_bbox[0]-x_min,cur_bbox[1]-y_min),(cur_bbox[2]-x_min,cur_bbox[3]-y_min),color_inter_mask,2) 
    cv2.rectangle(label_im_box,(0,0),(w_max,h_max),color_mBox,2)   #max_box
    ##inter_im=inter_label_im_box 
    cv2.putText(inter_label_im_box, 'intersection mask',(int(w_max/4),int(10)),font,0.5,(255,255,255),2)  

    # ## link image
    # link_im= np.zeros((h_max, w_max*5, 3), dtype=np.uint8)    
    # link_im[:,0:w_max,:]=ft_label_im_box
    # link_im[:,w_max:w_max*2,:]=label_im_box
    # link_im[:,w_max*2:w_max*3,:]=bt_label_im_box
    # link_im[:,w_max*3:w_max*4,:]=inter_label_im_box
    # link_im[:,w_max*4:w_max*5,:]=pk_label_im_box 
    link_im= np.zeros((h_max, w_max*2, 3), dtype=np.uint8)  
    link_im[:,0:w_max,:]=label_im_box
    link_im[:,w_max:w_max*2,:]=pk_label_im_box 
    return link_im

## Continue from here...
def vis_picked_masks(iter_label_dir,rgb_ims, frm_ids, y_labels,picked_masks,picked_bboxes,fw_masks,prev_boxes,cur_masks,cur_boxes,bw_masks,next_boxes):
    set_name=mcfg.DATA.SEQNAME
    frm_num=len(frm_ids)
    ##vis and save mask.
    print 'vis picked masks...'
    for n_id in xrange(frm_num):
        im_name=set_name+'_'+str(frm_ids[n_id]).zfill(6)
        vis_im_path=os.path.join(iter_label_dir,im_name+im_ext)  ## mask results from MRCNN
        im=rgb_ims[n_id]
        img_bbox=[0, 0, im.shape[1]-1,im.shape[0]-1]
        vis_link_im=vis_link_masks_im(fw_masks[n_id],prev_boxes[n_id],cur_masks[n_id],cur_boxes[n_id],bw_masks[n_id],next_boxes[n_id],picked_masks[n_id],picked_bboxes[n_id],y_labels[n_id],im,img_bbox) 
        cv2.imwrite(vis_im_path,vis_link_im)

def vis_final_masks():
    print 'vis final masks...'
    ## for calling final (comparison) visualization     
    ''' tmp visualization
  cur_masks=np.zeros((20,719,339))
            cur_masks[0]=picked_masks[0]
            cur_masks[1:-1,:,:]=picked_masks
            cur_masks[-1,:,:]=picked_masks[-1]
            cur_boxes=max_boxes
            
            if iter_id==iter_num-1:
                final_picked_masks=picked_masks
 
        print 'init_max_masks.shape:', init_max_masks.shape
        print 'final_picked_masks.shape:',final_picked_masks.shape
        print 'iou_arr:',iou_arr

        ##========================Vis final link mask====================================        
        final_label_dir=os.path.join(track_mask_label_dir,'final')
        create_dir(final_label_dir)

        for f_id in xrange(test_frm_num-2):
            mask1=init_max_masks[f_id]
            mask2=final_picked_masks[f_id]
            im=rgb_ims[f_id+1] ##due to not considering the first frame
            
            im_name=set_name+'_'+str(int(frm_ids[f_id+1])).zfill(6)
            im_path=os.path.join(final_label_dir,im_name+im_ext)
            vis_final_link_mask(mask1,mask2,im,max_bbox,im_path)         
             '''  