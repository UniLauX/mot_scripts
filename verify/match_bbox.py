## This function is for verify that match_box func is correct.(even in the edge cases)
import _init_paths
import io_mot as io
import load_mot as ld
from match_box import get_bboxes_center_shift,correct_shift_vec
from tud_crossing import load_dets_new
from det_util import dets_filter_frms,dets_filter_ids,parse_dets1
from common_lib import create_dir
from vis_box import vis_match_boxes

from mot_config import mcfg
import numpy as np
import os
import cv2

debug_flag=True
##[para 1]  set up all the parameters in this file
def parse_args():
    # 0> parameters
    start_frm_id=1
    s_frm=1
    e_frm=201
    s_index=s_frm-start_frm_id
    e_index=e_frm-start_frm_id
    frm_ids=np.arange(s_frm,e_frm+1)
    #obj_ids=[5]
    obj_ids=np.array([5, 10, 12, 13, 14])
    return frm_ids, obj_ids, start_frm_id

##[para 2] for saving the match_box results
def get_match_res_dir(obj_id):
    ### create gt_masks_dir, corrup_masks_dir and pick_masks_dir for later use
    print '=========================get match res dir====================='
    # seg_dir
    seg_dir=mcfg.EXPR.SEG_DIR 
    dataset=mcfg.DATA.DATASET
    seg_algr=mcfg.EXPR.SEG_ALGR
    seg_dir=os.path.join(seg_dir,dataset,seg_algr)

    # random_corrup_dir
    rand_corrup_dir=os.path.join(seg_dir,'random_corrup')
    create_dir(rand_corrup_dir)
    
    ##match_boxes_dir
    match_boxes_dir=os.path.join(rand_corrup_dir,'match_boxes')
    create_dir(match_boxes_dir)

    match_boxes_dir=os.path.join(match_boxes_dir,str(obj_id))
    create_dir(match_boxes_dir)

    return match_boxes_dir

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

## filter dets
def get_filter_gt_annots(frm_ids,obj_ids):
    ##print '=========get_filter_gt_annots================='
    ## need to interplate to make the bboxes be continious
    gt_dets=load_dets_new()
    gt_dets,row_indexes1=dets_filter_frms(gt_dets,frm_ids)
    gt_dets,row_indexes2=dets_filter_ids(gt_dets,obj_ids) 
    return gt_dets

## main func
if __name__ == '__main__':
    ##continious labels: 5, 10, 12, 13, 14
    print '=================verify match bboxes===================='
    filt_frm_ids, filt_obj_ids, start_frm_id=parse_args()
    frm_indexes=filt_frm_ids-start_frm_id

    # 1> im_names and ims
    im_names=get_im_names()          # whole
    im_names=im_names[frm_indexes]   
    
    rgb_ims=load_color_ims(im_names)

    for f_obj_id in filt_obj_ids[4:5]:
        ##gt_boxes
        gt_dets=get_filter_gt_annots(filt_frm_ids,[f_obj_id])
        gt_frm_ids,gt_obj_ids,gt_boxes=parse_dets1(gt_dets) # whole 
        gt_boxes=gt_boxes.astype(int)
        
        # print '------------------------------------------'
        # if debug_flag:
        #     print 'filt_obj_ids:', filt_obj_ids
        #     print 'im_names:',im_names
        #     print 'gt_dets:',gt_dets 
        ## res_dir
        match_boxes_dir=get_match_res_dir(f_obj_id)

        ## filter im_names and ims(acorrding to filtered frm_ids)
        flt_indexes=gt_frm_ids-start_frm_id
        
        print '--------------------------------------------------',f_obj_id
        print 'gt_frm_ids:', gt_frm_ids
        print 'flt_indexes:', flt_indexes

        f_im_names=im_names[flt_indexes]

        ims=np.asarray(rgb_ims)
        cur_ims=ims[flt_indexes]
        # (match_boxes) shift vec
        shift_vec=get_bboxes_center_shift(cur_ims,gt_boxes)

        shift_vec=correct_shift_vec(shift_vec)
        ## trick: better to replace with algorithm
        if f_obj_id==10:
            shift_vec[8]=(2/3.0)*shift_vec[7]+(1/3.0)*shift_vec[10]
            shift_vec[9]=(1/3.0)*shift_vec[7]+(2/3.0)*shift_vec[10] 
            shift_vec[21]=(2/3.0)*shift_vec[20]+(1/3.0)*shift_vec[23]
            shift_vec[22]=(1/3.0)*shift_vec[20]+(2/3.0)*shift_vec[23] 
        shift_vec=shift_vec.astype(int)    
        
        # print 'shift_vec:',np.where(shift_vec[:,0]>10)[0]
        # print 'shift_vec:',np.where(shift_vec[:,0]<-10)[0]
        # print 'shift_vec:',np.where(shift_vec[:,1]>10)[0]
        # print 'shift_vec:',np.where(shift_vec[:,1]<-10)[0]
        print 'shift_vec:', shift_vec
        ## per-obj
        vis_match_boxes(match_boxes_dir,f_im_names,cur_ims,gt_boxes,shift_vec)

        '''
        if debug_flag:
            print 'len(im_names):', len(im_names)
            print 'len(rgb_ims):', len(cur_ims)
            print 'gt_dets.shape:', gt_dets.shape
            print 'gt_boxes.shape:', gt_boxes.shape
            print 'match_boxes_dir:', match_boxes_dir
            print 'shift_vec:', shift_vec
        
        
        print 'shift_vec:', shift_vec
        '''
    ## In sum: some match_boxes are totally wrong...
    ##  not continious:
    # random_corrup: frm 57~61 has some problems
    # 92 is totally wrong
    ##nearby the border shift_vec are 0, start from 86 
    ## gt_boxes are not correct at 17, 30