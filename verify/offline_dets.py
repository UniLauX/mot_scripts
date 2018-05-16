import _init_paths
from mot_config import mcfg
import io_mot as io
import load_mot as ld
import numpy as np
import os

from det_util import form_dets1
from test_vis import load_frame_boxes,load_frame_masks
from mot_util import write_detarr_as_mot
from box_util import coords_to_boxes  
from mask_util import masks_align_boxes
from common_lib import create_dir,load_txt_to_fltarr
debug_flag=True

##parameters
seq_name=mcfg.DATA.SEQNAME
shift_num=mcfg.DATA.SEQ_SHIFT
iter_num=mcfg.FTRACK.ITER_NUM
iter_name='iter0'
##iter_name='iter_conv'   ## org
im_width=640
im_height=480
#-------------------------------dir----------------------------------------------------------
## create ref_dir for saving dets_arr and masks_tensor
def get_ref_dir():
    track_mask_dir=io.get_track_mask_info(seq_name)
    refine_res_dir=os.path.join(track_mask_dir,'res')
    create_dir(refine_res_dir)
    return refine_res_dir

## track_mask_dir is the root folder for 'pick masks'
def get_track_mask_dir():
    track_mask_dir=io.get_track_mask_info(seq_name)
    return track_mask_dir

## prepare continious dets(as input for 'pick-mask' algorithm)
def get_pre_cont_dets():
    # print 'prop_file_path:', prop_file_path
    # print 'prop_file_ext:', prop_file_ext
    prop_file_path,prop_file_ext=io.get_dets_info(seq_name)
    prop_arr=ld.load_det_proposals(prop_file_path)
    return prop_arr

#------------------------------dets----------------------------------------------------
## transfer the refine('pick-mask') dets, from local disk to dets_arr 
def get_ref_dets(track_mask_dir,prop_arr):
    print '================get_ref_dets==========================='
    # if debug_flag:
    #     print 'track_mask_dir:', track_mask_dir
    #     print 'prop_arr.shape:', prop_arr.shape

    det_frm_ids=np.empty((0,),dtype=int)
    det_obj_ids=np.empty((0,),dtype=int)

    det_boxes=np.empty((0,4))
    uniq_frm_ids=np.sort(np.unique(prop_arr[:,0]).astype(int))
    frm_num=len(uniq_frm_ids)

    ## calculate based on per-frame
    for n_id in xrange(frm_num):
    ##for n_id in xrange(2):
        frm_id=uniq_frm_ids[n_id]
        row_indexs=np.where(prop_arr[:,0]==frm_id)[0]      ##frame indexes

        frm_prop_arr=prop_arr[row_indexs]
        frm_labels=frm_prop_arr[:,1].astype(int)
        frm_ids=frm_prop_arr[:,0] 
        
        det_frm_ids=np.concatenate((det_frm_ids,frm_ids))
        det_obj_ids=np.concatenate((det_obj_ids,frm_labels))
        
        boxes=load_frame_boxes(track_mask_dir,frm_id,frm_labels,iter_name)         
        det_boxes=np.concatenate((det_boxes,boxes), axis=0) 

    ref_dets=form_dets1(det_frm_ids,det_obj_ids,det_boxes)

    # if debug_flag:
    #     print 'det_frm_ids:', det_frm_ids.shape
    #     print 'det_obj_ids:', det_obj_ids.shape
    #     print 'ref_dets:', ref_dets.shape
    #     print 'track_mask_dir:', track_mask_dir
    return ref_dets

## save dets_arr in local disk
def save_ref_dets(res_dir,dets_arr):
    dets_path=os.path.join(res_dir,'ref_dets.txt')
    write_detarr_as_mot(dets_path,dets_arr)

## load dets that with tight_gt_boxes
def load_dets_ref():
    res_dir=get_ref_dir()
    dets_path=os.path.join(res_dir,'ref_dets.txt')
    dets=load_txt_to_fltarr(dets_path)
    return dets

#---------------------------------masks---------------------------------------------------------
## transfer the refine(pick-mask) binary masks in one masks_tensor and save it in local disk
def get_ref_masks(track_mask_dir,prop_arr):  ## masks has the same size with the boxes
    print '================get_ref_masks==========================='

    det_masks=[]

    if debug_flag:
        print 'track_mask_dir:', track_mask_dir
        # print 'prop_file_path:', prop_file_path
        # print 'prop_file_ext:', prop_file_ext
        print 'prop_arr.shape:', prop_arr.shape
        
    uniq_frm_ids=np.sort(np.unique(prop_arr[:,0]).astype(int))
    frm_num=len(uniq_frm_ids)

    ## calculate based on per-frame
    for n_id in xrange(frm_num):
    ##for n_id in xrange(2):
        frm_id=uniq_frm_ids[n_id]
        row_indexs=np.where(prop_arr[:,0]==frm_id)[0]      ##frame indexes
        frm_prop_arr=prop_arr[row_indexs]

        frm_labels=frm_prop_arr[:,1].astype(int) 

        im_name=seq_name+'_'+str(uniq_frm_ids[n_id]+shift_num).zfill(6)   ##im_name

        masks=load_frame_masks(track_mask_dir,frm_labels,iter_name,im_name) 
        #arr_masks=np.asarray(masks)
        # print len(masks)
        # # print 'type(det_masks):', type(det_masks)
        # # print 'type(arr_masks):', type(arr_masks)
        # print 'det_masks.shape:', det_masks.shape
        # print 'arr_masks.shape:', arr_masks.shape
        det_masks=det_masks+masks
        # print len(det_masks)
    det_masks=np.array(det_masks)    
        # det_masks=np.concatenate((det_masks,arr_masks)) 
    return det_masks   # eg. shape=(1214,)

## get masks which has the same size as color image(from mini_box_masks)
def get_enlarge_masks(track_mask_dir,cont_dets):
    ## mini_masks for ref-masks
    mini_masks=get_ref_masks(track_mask_dir,cont_dets)

    # transfer small masks(same size with boxes) to large masks(same size with color image)
    mask_num=len(mini_masks)
    masks=np.zeros((mask_num,im_height,im_width),dtype=int)

    ## mini_boxes
    dets=get_ref_dets(track_mask_dir,cont_dets) 
    coords=dets[:,2:6]
    boxes=coords_to_boxes(coords)
    boxes=boxes.astype(int)
 
    ## max_boxes
    max_boxes=np.zeros((mask_num,4))
    max_boxes[:,2]=im_width-1
    max_boxes[:,3]=im_height-1
    max_boxes=max_boxes.astype(int)

    ## max_masks
    max_masks=masks_align_boxes(mini_masks,boxes,max_boxes) 
    arr_masks=np.asarray(max_masks)
    return arr_masks

## save ref_masks_tensor in local disk
def save_ref_masks(res_dir,masks_tensor):
    masks_path=os.path.join(res_dir,'ref_masks')
    print masks_path
    print masks_tensor.shape
    np.save(masks_path,masks_tensor)


## load ref_masks_tensor from local disk
def load_masks_ref():
    res_dir=get_ref_dir()
    masks_path=os.path.join(res_dir,'ref_masks.npy')
    print masks_path
    masks_tensor=np.load(masks_path)
    return masks_tensor

if __name__ == '__main__':
    print '===================offline dets=================='
    ## res dir
    res_dir=get_ref_dir()

    ## track_mask_dir and cont_dets
    track_mask_dir=get_track_mask_dir()
    cont_dets=get_pre_cont_dets()

    ## save dets in .txt
    ref_dets=get_ref_dets(track_mask_dir,cont_dets)

    save_ref_dets(res_dir,ref_dets)

    ## save masks in .npy   
    ref_masks_tensor=get_enlarge_masks(track_mask_dir,cont_dets)
    
    save_ref_masks(res_dir,ref_masks_tensor)
    
    if debug_flag:
        print 'res_dir:', res_dir
        print 'ref_dets.shape:', ref_dets.shape 
        print 'ref_masks_tensor.shape:', ref_masks_tensor.shape
