##function description: using pariwise optical flow to track the noisy masks 

import _init_paths
import numpy as np
import os 
import cv2
import cPickle
from mot_config import mcfg
from common_lib import create_dir,color_map ##load_color_images,load_txt_to_arr,
from flowlib import read_flo_file,visualize_flow
# from match_box import color_diff_matching as bf_match_box
# from match_box import match_template as mt_match_box
from match_box import get_bboxes_center_shift

from pick_mask import lpsolve_pick_mask
from visual_masks import vis_propagated_masks
from visual_masks import vis_picked_masks
from vis_mask import  vis_final_link_mask

import io_mot as io
import load_mot as ld
from box_util import coords_to_boxes,boxes_to_coords,boxes_align_masks
from mask_util import load_binary_mask
from det_util import dets_filter_frms, dets_filter_ids,parse_dets1

from generate_util import save_picked_masks
from generate_util import save_picked_proposals

from flow_track import fw_warp_prop_mask,bw_warp_prop_mask  ## forward_warp and backward_warp to propogate mask

from match_box import correct_shift_vec

# switches
vis_propagate_flag=False
vis_picked_flag=True
vis_final_flag=False
save_masks_flag=True  ## save picked masks
save_dets_flag=True   ## save picked boxes(as well as obj_id and frm_ids)
debug_flag=True

def cal_per_label(rgb_ims,forward_flows,backward_flows,frm_ids,bboxes,masks,track_way='bw_warp'):
    ## parameters
    set_name=mcfg.DATA.SEQNAME
    im_ext=mcfg.DATA.IMGEXT
    mask_im_ext=mcfg.MASK.BIN_IM_EXT

    ## tracked masks
    fw_masks=[]
    bw_masks=[]

    ##tight bboxes
    fw_track_bboxes=[]
    bw_track_bboxes=[]

    frm_num=len(frm_ids)
    ##for n_id in xrange(0,3):   ##testing
    for n_id in xrange(frm_num):
        im_name=set_name+'_'+str(frm_ids[n_id]).zfill(6)

        ##(1) load image
        im=rgb_ims[n_id]

        ##(2) vis noisy_mask(current frame)
        label_mask=masks[n_id]
        bbox=bboxes[n_id]

        ##(3) get forward tracked mask----------------------------------------------------------
        if n_id==0:        ##special case for the first frame(copy mask-rcnn for fw-track-mask)
            fw_track_bbox=bbox
            fw_track_mask=label_mask.copy()
        else:
            prev_im_name=set_name+'_'+str(frm_ids[n_id-1]).zfill(6)
            prev_label_mask=masks[n_id-1]
            if track_way=='fw_warp':                     ## forward warp prev_mask
                prev_fw_flow=forward_flows[n_id-1]
                prev_bbox=bboxes[n_id-1]
                fw_track_bbox,fw_track_mask=fw_warp_prop_mask(prev_label_mask,prev_fw_flow,prev_bbox)  #
            else:                                        ## backward warp next_mask
                bw_flow=backward_flows[n_id-1]
                prev_bbox=bboxes[n_id-1]
                fw_track_bbox,fw_track_mask=bw_warp_prop_mask(prev_label_mask,bw_flow,prev_bbox)  # 
        
        ##(4) get backward tracked mask-----------------------------------------------------------
        if n_id==frm_num-1:       ##special case for the last frame(copy mask-rcnn for bw-track-mask)
            bw_track_bbox=bbox
            bw_track_mask=label_mask.copy()
        else:
            next_im_name=set_name+'_'+str(frm_ids[n_id+1]).zfill(6)
            next_label_mask=masks[n_id+1]
            if track_way=='fw_warp':
                bw_flow=backward_flows[n_id]  #backflow was named with current frame
                next_bbox=bboxes[n_id+1]   
                bw_track_bbox,bw_track_mask=fw_warp_prop_mask(next_label_mask,bw_flow,next_bbox) 
            else:
                prev_fw_flow=forward_flows[n_id]
                next_bbox=bboxes[n_id+1]  
                bw_track_bbox,bw_track_mask=bw_warp_prop_mask(next_label_mask,prev_fw_flow,next_bbox)  # 
        fw_masks.append(fw_track_mask)
        bw_masks.append(bw_track_mask)

        fw_track_bboxes.append(fw_track_bbox)
        bw_track_bboxes.append(bw_track_bbox)

    ## new produce bboxes by tight propagated masks
    fw_track_bboxes_arr=np.asarray(fw_track_bboxes)
    bw_track_bboxes_arr=np.asarray(bw_track_bboxes)
    return fw_track_bboxes_arr,bw_track_bboxes_arr,fw_masks, bw_masks  

'''    
def get_bboxes_center_shift(ims,bboxes,motion_estm_meth):
    im_num=len(ims)
    box_num=len(bboxes)
    
    if im_num!=box_num:
        print 'image number is not equal to box number...'
        return
    if im_num<2:
        print 'image number is less than 2, could not do bboxes_center_shift...'
        return   
    vec_len=box_num-1
    shift_vec=np.zeros((vec_len,2),dtype=float)  ##(x01,y01),(x12,y12),(x23,y23)
    
    if motion_estm_meth=='match_template':
        print motion_estm_meth,'...'
        for s_id in xrange(vec_len):
             swap_flag, map_bbox,bbox_center_shift=mt_match_box(ims[s_id],ims[s_id+1],bboxes[s_id],bboxes[s_id+1])
             shift_vec[s_id]=bbox_center_shift

    elif motion_estm_meth=='bruteforce':
        print 'bruteforce...'
        for s_id in xrange(vec_len):
             swap_flag, map_bbox,bbox_center_shift=bf_match_box(ims[s_id],ims[s_id+1],bboxes[s_id],bboxes[s_id+1])
             shift_vec[s_id]=bbox_center_shift
    else:## offline by bruthforce
        print 'no motion considering...'
          #shift_vec=[[ 4. , 0.],[ 5. , 0.],[ 5. , 0.],[ 5. , 1.],[ 6. , 0.],[ 5. , 1.],[ 6. , 0.],[ 7. , 0.],[ 8. , 0.],[ 7. , 0.],[ 9. , 0.],[ 8. , 0.],[ 9. , 0.],[ 8. , 0.],[ 8. , 0.],[ 8. , 0.],[ 9. , 0.],[ 8. , 0.],[ 8. , 0.]]  
    return shift_vec
    '''

## get frm_info (args1)
def get_seq_info():
    ##(0) parameters(set-specific paras)
    test_frm_num=mcfg.DATA.SEQ_LENGTH  # eg.20(org)
    gt_start_index=mcfg.DATA.SEQ_GT_START_INDEX
    shift_num=mcfg.DATA.SEQ_SHIFT  ## =0 by default
    
    start_frm_id=gt_start_index+shift_num   ## usually start index is 1
    end_frm_id=(gt_start_index+test_frm_num-1)+shift_num
    return test_frm_num,gt_start_index,shift_num,start_frm_id,end_frm_id  

## parse args(args2)
def parse_args():
    motion_estm_meth='match_template' 
    dataset=mcfg.DATA.DATASET          ##eg. #MOT17
    seq_name=mcfg.DATA.SEQNAME 
    iter_num=mcfg.FTRACK.ITER_NUM      ##iter times (note: iter_num should not fixed for different instances) 
    track_mask_dir=io.get_track_mask_info(seq_name)  ## output dir 
    return motion_estm_meth,dataset,seq_name,iter_num,track_mask_dir

## get im_names
def get_im_names(start_index,end_index):
    imgset_path,jpg_dir,im_ext=io.get_ims_info()   
    im_names=ld.get_im_names(imgset_path)            ##get all the im_names(always)
    im_names=im_names[start_index:end_index+1]    ##trick2
    if debug_flag:
        print 'imgset_path:',imgset_path
        print 'jpgdir_path:',jpg_dir
        print 'im_ext:',im_ext
    return im_names

## load color ims according to im_names
def load_color_ims(im_names):
    imgset_path,jpg_dir,im_ext=io.get_ims_info()   
    rgb_ims=ld.load_color_images(jpg_dir,im_names,im_ext)
    return rgb_ims

## load forward(and backward) optical flow files
def load_opt_flows(seq_name,im_names):
    optflow_dir,flow_im_ext=io.get_flows_info(seq_name)
    forward_flows,backward_flows=ld.load_opticalflows_flo(optflow_dir,im_names,flow_im_ext)
    if debug_flag:
        print 'flow dir:', optflow_dir
        print 'flow_im_ext:',flow_im_ext   
    return forward_flows,backward_flows

## load all the dets, need to be filtered
def load_det_proposals(seq_name,start_frm_id,end_frm_id):
    prop_file_path,prop_file_ext=io.get_dets_info(seq_name)
    prop_arr=ld.load_det_proposals(prop_file_path)    ## load all the proposals(always)
    frm_ids=np.arange(start_frm_id,end_frm_id+1)
    prop_arr,row_indexes=dets_filter_frms(prop_arr,frm_ids)
    if debug_flag:
        print 'prop_file_path:', prop_file_path
        print 'prop_file_ext:', prop_file_ext
    return prop_arr

## load masks according to dets(order)
def load_seg_proposals(seq_name,prop_arr):
    mask_dir,mask_im_ext=io.get_segs_info(seq_name)
    det_num=len(prop_arr)
    if det_num<1:
        print 'det_num is less than 1...'
        return

    list_init_masks=[]
    for n_id in np.arange(det_num):
        row_det=prop_arr[n_id].astype(int)
        frm_id, obj_id=row_det[:2]
        im_name=seq_name+'_'+str(frm_id).zfill(6)
        label_mask_path=os.path.join(mask_dir,str(obj_id),im_name+mask_im_ext)
        label_mask=load_binary_mask(label_mask_path)
        ##print 'label_mask_path:', label_mask_path
        list_init_masks.append(label_mask)
        init_masks=np.asarray(list_init_masks) ##cur_masks has different shape  
    return init_masks
    
## amend boxes according to masks' shape(from float to int)
def amend_dets(masks,in_dets):
    out_dets=in_dets
    in_coords=in_dets[:,2:6]
    in_boxes=coords_to_boxes(in_coords)
    out_boxes=boxes_align_masks(masks,in_boxes) 
    out_coords=boxes_to_coords(out_boxes)
    out_dets[:,2:6]=out_coords
    return out_dets

## main entry of refine mask func
def refine_masks(track_mask_dir,rgb_ims,forward_flows,backward_flows,masks,prop_arr,obj_ids=[],start_frm_id=1):
    
    ##parameters:
    motion_estm_meth='match_template' 

    ##get filtered dets and masks according to (filter) frm_ids and obj_ids 
    if len(obj_ids)==0:
        labels=np.unique(prop_arr[:,1]).astype(int)
    else:
        labels=obj_ids
        prop_arr,row_indexes0=dets_filter_ids(prop_arr,obj_ids)
        masks=masks[row_indexes0]

    if debug_flag:
        print 'labels:', labels
        print 'prop_arr.shape:', prop_arr.shape

    for label in labels:
        lab_prop_arr,row_indexes1=dets_filter_ids(prop_arr,np.array([label]))
        init_masks=masks[row_indexes1]
        frm_ids,obj_ids,boxes=parse_dets1(lab_prop_arr)
        init_boxes=boxes.astype(int)

        if len(frm_ids)==0:                           ## due to 'filter' conditions, len(frm_id) may =0 for some 'labels'
            continue

        cur_indexes=frm_ids-start_frm_id
        arr_ims=np.asarray(rgb_ims)
        cur_ims=arr_ims[cur_indexes]
        
        cur_fw_flows=forward_flows[cur_indexes[:-1]]
        cur_bw_flows=backward_flows[cur_indexes[:-1]]
       
        shift_vec=get_bboxes_center_shift(cur_ims,init_boxes,motion_estm_meth)   ##shift_vec.shape=(frm_num-1,2) 
        shift_vec=correct_shift_vec(shift_vec)

        ## trick(remove it later)
        if label==10:
            shift_vec[8]=(2/3.0)*shift_vec[7]+(1/3.0)*shift_vec[10]
            shift_vec[9]=(1/3.0)*shift_vec[7]+(2/3.0)*shift_vec[10] 
            shift_vec[21]=(2/3.0)*shift_vec[20]+(1/3.0)*shift_vec[23]
            shift_vec[22]=(1/3.0)*shift_vec[20]+(2/3.0)*shift_vec[23] 
        shift_vec=shift_vec.astype(int)    
        
        print 'shift_vec:', shift_vec      ## shift_vec is not robust at present
        ## online pick iteration(func)
        online_pick_iteration(track_mask_dir,label,frm_ids,init_masks,init_boxes,cur_ims,cur_fw_flows,cur_bw_flows,shift_vec)
    
## on-line propogate and track mask. very slow, a better way is save all the files off-line in advance and using indexes
def online_pick_iteration(track_mask_dir,label,frm_ids,init_masks,init_boxes,cur_ims,cur_fw_flows,cur_bw_flows,shift_vec):
    print 'online pick iteration...'
    ## para0: max_bbox
    iter_num=mcfg.FTRACK.ITER_NUM
    seq_name=mcfg.DATA.SEQNAME
    im_ext=mcfg.DATA.IMGEXT
    mask_im_ext=mcfg.MASK.BIN_IM_EXT
    
    track_lab_mask_dir=os.path.join(track_mask_dir,str(label))
    create_dir(track_lab_mask_dir)

    if debug_flag:
        print 'track_mask_dir:', track_mask_dir
        print 'track_lab_mask_dir:', track_lab_mask_dir
        print 'iter_num:', iter_num

    iou_arr=np.zeros(iter_num)
    cur_masks=init_masks
    cur_boxes=init_boxes
    
    conv_flag=False
    iter_id=0   ## counter
    ave_iou=0.0
    old_ave_iou=0.0

    while iter_id<iter_num and conv_flag==False: 
    #for iter_id in xrange(iter_num):
        old_ave_iou=ave_iou

        picked_masks=[]
        picked_boxes=[]

        ## ---------------------------------------create dirs----------------------------------------------------------------------        
        ## iter dir
        iter_name='iter'+str(iter_id)
        iter_label_prop_path=os.path.join(track_lab_mask_dir,iter_name+'.txt') ##picked dets(boxes) file path
      
        ## picked masks dir
        iter_label_dir=os.path.join(track_lab_mask_dir,iter_name)   
        create_dir(iter_label_dir)
        
        ## vis for debuging
        vis_iter_name='vis_iter'+str(iter_id)
        vis_iter_label_dir=os.path.join(track_lab_mask_dir,vis_iter_name)
        create_dir(vis_iter_label_dir)
       
        ## new added, for converage  
        ## converage .txt
        iter_conv_name='iter_conv'
        iter_conv_label_prop_path=os.path.join(track_lab_mask_dir,iter_conv_name+'.txt') ##picked dets(boxes) file path      
        
        # # converage masks dir
        iter_conv_label_dir=os.path.join(track_lab_mask_dir,iter_conv_name)   
        create_dir(iter_conv_label_dir)
        
        ## pick masks in different cases
        if len(cur_ims)>2:
            prev_boxes,next_boxes,fw_masks,bw_masks=cal_per_label(cur_ims,cur_fw_flows,cur_bw_flows,frm_ids,cur_boxes,cur_masks) 
            
            picked_bboxes,picked_masks,y_labels,ave_iou=lpsolve_pick_mask(cur_ims,frm_ids,prev_boxes,cur_boxes,next_boxes,fw_masks,cur_masks,bw_masks,shift_vec) 
            
            if vis_picked_flag:
                vis_picked_masks(vis_iter_label_dir,cur_ims, frm_ids,y_labels,picked_masks,picked_bboxes,fw_masks,prev_boxes,cur_masks,cur_boxes,bw_masks,next_boxes)
        else:                      ## just one image
            picked_masks=cur_masks
            picked_bboxes=cur_boxes
       
        iou_arr[iter_id]=ave_iou

        ## judge if it's converaged
        if ave_iou==old_ave_iou:
            conv_flag=True
        else:
            cur_masks=picked_masks
            cur_boxes=picked_bboxes 
            
        if save_masks_flag:
            save_picked_masks(iter_label_dir,seq_name,picked_masks,frm_ids)   ## save binary masks(for later using)
            
        if save_dets_flag:
            save_picked_proposals(iter_label_prop_path,frm_ids,label,picked_bboxes) ## save det proposals

        print 'iou_arr:', iou_arr

        ## save conv results
        if conv_flag==True or iter_id==iter_num-1:
            if save_masks_flag:
                save_picked_masks(iter_conv_label_dir,seq_name,cur_masks,frm_ids)   ## save binary masks(for later using)
            if save_dets_flag:
                save_picked_proposals(iter_conv_label_prop_path,frm_ids,label,cur_boxes) ## save det proposals

        iter_id+=1 ## loop 
        
if __name__ == '__main__':
    print '===============================parameters setup==========================================='  
    obj_ids=[] ## parameters  
    motion_estm_meth,dataset,seq_name,iter_num,track_mask_dir=parse_args()
    test_frm_num,gt_start_index,shift_num,start_frm_id,end_frm_id=get_seq_info()
    start_index=start_frm_id-gt_start_index   ## index start from 0
    end_index=end_frm_id-gt_start_index
    # if debug_flag:
    #     print 'dataset:', dataset 
    #     print 'seq_name:', seq_name
    #     print 'iter_num:', iter_num
    #     print 'test_frm_num:', test_frm_num
    #     print 'shift_num:', shift_num
    #     print 'start_frm_id:',start_frm_id
    #     print 'end_frm_id:', end_frm_id
    #     print 'track_mask_dir:',track_mask_dir
    print '===========================================input============================================'
    im_names=get_im_names(start_index,end_index)
    rgb_ims=load_color_ims(im_names)
    forward_flows,backward_flows=load_opt_flows(seq_name,im_names)
    
    ## im_width and im_height
    im_width=rgb_ims[0].shape[1]
    im_height=rgb_ims[0].shape[0]
    max_bbox=[0,0,im_width-1, im_height-1]

    ## masks and dets(boxes match masks's shape)
    prop_arr=load_det_proposals(seq_name,start_frm_id,end_frm_id)
    masks=load_seg_proposals(seq_name,prop_arr)  # not finished(write all the masks into one tensor later)
    prop_arr=amend_dets(masks,prop_arr)

    if debug_flag:
        print 'im_width:', im_width
        print 'im_height:', im_height
        print 'prop_arr.shape:', prop_arr.shape
        print 'masks.shape:', masks.shape

    ##input: prop_arr,masks,obj_ids, forward_flows, backward_flows
    refine_masks(track_mask_dir,rgb_ims,forward_flows,backward_flows,masks,prop_arr,obj_ids) 