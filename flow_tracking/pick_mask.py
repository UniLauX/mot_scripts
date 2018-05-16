import _init_paths
from GraphCRF_demo import crf_infer as crf_if ##(not used)
from lpsolve_demo import  lpsolve_infer as lps_if
import cv2
import numpy as np
from mot_config import mcfg

def lpsolve_pick_mask(rgb_ims,frm_ids,prev_boxes,cur_boxes,next_boxes,fw_masks,cur_masks,bw_masks,shift_vec):
    seq_name=mcfg.DATA.SEQNAME
    im_ext=mcfg.DATA.IMGEXT
    
    frm_num=len(cur_masks)
    fw_frm_num=len(fw_masks)
    y_labels=np.zeros(frm_num)
    ave_iou=0.0
    picked_masks=[]
    picked_boxes=[]
    print 'lpsolve pick mask...'
    picked_boxes,picked_masks,y_labels,ave_iou=lps_if(rgb_ims,prev_boxes,cur_boxes,next_boxes,fw_masks,cur_masks,bw_masks,shift_vec)
    return picked_boxes,picked_masks,y_labels,ave_iou

## not used now,based on fw_mask.shape=cur_mask.shape=bw_mask.shape
def crf_pick_mask(fw_masks,cur_masks,bw_masks,rgb_ims,set_name,frm_ids,im_ext,max_bbox):
    print 'using crf to pick the best mask start...'
    frm_num=len(cur_masks)  ## doesn't include the first frame and last frame.
    fw_frm_num=len(fw_masks)
    im_num=len(rgb_ims)
    y,picked_masks=crf_if(fw_masks,cur_masks,bw_masks)

    for frm_id in xrange(frm_num):
        im_name=set_name+'_'+str(int(frm_ids[frm_id+1])).zfill(6)
        vis_im_path=os.path.join(iter_label_dir,im_name+im_ext)  ## mask results from MRCNN
        im=rgb_ims[frm_id+1]
        vis_link_im=vis_link_masks_im(fw_masks[frm_id],cur_masks[frm_id],bw_masks[frm_id],picked_masks[frm_id,:,:],y[frm_id],im,max_bbox) 
        cv2.imwrite(vis_im_path,vis_link_im)
        print 'im_name:',im_name
        print 'vis_im_path:',vis_im_path
    print 'y:',y
    print 'picked_masks.shape:',picked_masks.shape
    return y, picked_masks