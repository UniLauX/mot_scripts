## This file is for verifying the backward_warp for (optical flow)tracking.
## in forward_warp, when heavy occulution occurs, several pixels may be mapped to one pixel, thus cause the hole  
## compared with the forward_warp, backward_warp can fill the hole to some extent.
import _init_paths
import io_mot as io
import load_mot as ld
from tud_crossing import load_dets_new,load_masks_new
from det_util import dets_filter_frms,dets_filter_ids,parse_dets1
from flow_track import fw_warp_prev_mask,bw_warp_prev_mask
from mot_config import mcfg
from vis_mask import vis_alpha_mask
import numpy as np

seq_name=mcfg.DATA.SEQNAME
debug_flag=True
##----------------------------------------- gt info ----------------------------------------------------
## get im_names list without .ext
def get_im_names():
    imgset_path,jpg_dir,im_ext=io.get_ims_info()   
    im_names=ld.get_im_names(imgset_path)
    return im_names

## load color images based on im_names
def load_color_ims(im_names):
    imgset_path,jpg_dir,im_ext=io.get_ims_info()   
    ims=ld.load_color_images(jpg_dir,im_names,im_ext)  
    return ims

def get_filter_gt_annots(frm_ids,obj_ids):
    print '=========get_filter_gt_annots================='
    gt_dets=load_dets_new()
    gt_dets,row_indexes1=dets_filter_frms(gt_dets,frm_ids)
    gt_dets,row_indexes2=dets_filter_ids(gt_dets,obj_ids)

    gt_masks_tensor=load_masks_new()
    gt_masks=gt_masks_tensor[row_indexes1,:,:]
    gt_masks=gt_masks[row_indexes2,:,:]
    # print 'row_indexes1:', row_indexes1
    # print 'row_indexes2:', row_indexes2
    return gt_dets,gt_masks

##  set up all the parameters in this file
def parse_args():
    # 0> parameters
    start_frm_id=1
    s_frm=1
    e_frm=2
    s_index=s_frm-start_frm_id
    e_index=e_frm-start_frm_id
    
    frm_ids=np.arange(s_frm,e_frm+1)
    ##frm_indexes=frm_ids-start_frm_id
    obj_ids=[2]
    return frm_ids, obj_ids, start_frm_id

    
if __name__ == '__main__':
    print '=================verify backward_warp===================='
    frm_ids, obj_ids, start_frm_id=parse_args()
    frm_indexes=frm_ids-start_frm_id 

    # 1> im_names and ims
    im_names=get_im_names()          # whole
    im_names=im_names[frm_indexes]   
    rgb_ims=load_color_ims(im_names) 

    # 2> load optical flows
    optflow_dir,flow_im_ext=io.get_flows_info(seq_name)
    forward_flows,backward_flows=ld.load_opticalflows_flo(optflow_dir,im_names,flow_im_ext)

    # 3> gt masks and dets
    gt_dets,gt_masks=get_filter_gt_annots(frm_ids,obj_ids)
    gt_frm_ids,gt_obj_ids,gt_boxes=parse_dets1(gt_dets) # whole

    # 4> get two succesive images for testing
    prev_mask=gt_masks[0]
    fw_flow=forward_flows[0]
    bw_flow=backward_flows[0]
    prev_box=gt_boxes[0]

    # 5> call forward_warp and backward_warp   
    fw_warp_mask=fw_warp_prev_mask(prev_mask,fw_flow,prev_box)
    bw_warp_mask=bw_warp_prev_mask(prev_mask,bw_flow,prev_box)

    # 6> vis alpha_mask(result)
    vis_alpha_mask(prev_mask) 
    vis_alpha_mask(fw_warp_mask)
    vis_alpha_mask(bw_warp_mask)
    print fw_warp_mask.shape

    '''
    if debug_flag:
        # print 'frm_indexes:', frm_indexes 
        # print 'frm_ids:',frm_ids
        # print 'gt_dets.shape:', gt_dets.shape
        print 'gt_masks.shape:', gt_masks.shape
        print 'gt_frm_ids:', gt_frm_ids
        print 'gt_obj_ids:',gt_obj_ids
        print 'gt_boxes:', gt_boxes
        '''
   






    
    


