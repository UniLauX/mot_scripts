import _init_paths
import numpy as np
import os
import io_mot as io
import load_mot as ld
from vis_util import vis_proposals_multi_instances,vis_gt_bboxes
from mask_util import load_binary_mask
from common_lib import load_txt_to_fltarr
from box_util import coords_to_boxes
from vis_util import vis_link_imgs
from mot_config import mcfg
from tud_crossing import get_dets_annots_path,load_dets_annots

##set_name='MOT17-02'
set_name='TUD-Crossing'
track_mask_folder='track_mask'
test_frm_num=201
vis_mult_prop_folder='vis_mult_proposals'
iter_num=5
shift_num=0
debug_flag=True
mask_im_ext='.sm'
det_file_ext='.txt'

def load_frame_masks(track_mask_dir,frm_labels,iter_name,im_name):
    ##print '=====================load label masks======================='
    bin_masks=[]
    for n_id in xrange(len(frm_labels)):
        label=frm_labels[n_id]
        mask_im_path=os.path.join(track_mask_dir,str(label),iter_name,im_name+mask_im_ext)
        bin_mask=load_binary_mask(mask_im_path)
        bin_masks.append(bin_mask)    
    return bin_masks

def load_frame_boxes(track_mask_dir,frm_id,frm_labels,iter_name):
    ##print '======================load frame boxes==============================='
    frm_num=len(frm_labels)
    boxes=np.zeros((frm_num,4))
    for n_id in xrange(frm_num):
        label=frm_labels[n_id]
        det_file_path=os.path.join(track_mask_dir,str(label),iter_name+det_file_ext)
        ##print 'det_file_path:', det_file_path
        dets=load_txt_to_fltarr(det_file_path)
        r_frm_id=frm_id+shift_num
        row_index=np.where(dets[:,0]==r_frm_id)[0]
        boxes[n_id]=dets[row_index,2:6]
    return boxes

def vis_picked_mult_instances():
    print '=============================test vis_util================================='
    track_mask_dir=io.get_track_mask_info(set_name)
    prop_file_path,prop_file_ext=io.get_dets_info(set_name)
    prop_arr=ld.load_det_proposals(prop_file_path)
    
    if debug_flag:
        print 'track_mask_dir:', track_mask_dir
        print 'prop_file_path:', prop_file_path
        print 'prop_file_ext:', prop_file_ext
        print 'prop_arr.shape:', prop_arr.shape

    ##0> load image names  
    imgset_path,jpg_dir,im_ext=io.get_ims_info()   
    im_names=ld.get_im_names(imgset_path)

    im_names=im_names[:test_frm_num]   

    ##1> load color images
    rgb_ims=ld.load_color_images(jpg_dir,im_names,im_ext)
    vis_proposals_dir=io.get_vis_dir_info(vis_mult_prop_folder)
    iter_name='iter'+str(4)

    ## draw bbox per person 
    frm_ids=np.unique(prop_arr[:,0]).astype(int)
    frm_num=len(frm_ids)

    ##for debug
    for n_id in xrange(frm_num):  ## label is obj_id  (42 has some problems, 56)
        frm_id=frm_ids[n_id]
        row_indexs=np.where(prop_arr[:,0]==frm_id)[0]      ##frame indexes
        frm_prop_arr=prop_arr[row_indexs]
        frm_labels=frm_prop_arr[:,1].astype(int)
        im=rgb_ims[n_id]
        im_name=set_name+'_'+str(frm_ids[n_id]+shift_num).zfill(6)
        if len(frm_labels)==0:
            continue      

        masks=load_frame_masks(track_mask_dir,frm_labels,iter_name,im_name)
        boxes=load_frame_boxes(track_mask_dir,frm_id,frm_labels,iter_name)
        if debug_flag: 
            print 'frm_id:', frm_id
            print 'frm_labels:', frm_labels.shape
            print 'masks.shape:', len(masks)
            print 'boxes.shape:', boxes.shape
        vis_proposals_multi_instances(vis_proposals_dir,iter_name,im,im_name,boxes,masks,frm_labels)
       
    if debug_flag:
        print 'track_mask_dir:', track_mask_dir
        ##print 'im_names:', im_names
        print 'vis_proposals_dir:', vis_proposals_dir
        ##print 'boxes.shape:', boxes.shape


## for vis-comparing org-maskrcnn and refine-maskrcnn
def generate_final_vis_imgs():
    ##vis_picked_mult_instances()  ## run this first, and then the following...\
    final_vis_dir='/mnt/phoenix_fastdir/experiments/final_vis'
    final_vis_dir=os.path.join(final_vis_dir,set_name)
    org_mrcnn_dir=os.path.join(final_vis_dir,'org_MRCNN')
    pickd_mrcnn_dir=os.path.join(final_vis_dir,'picked_MRCNN')
    link_mrcnn_dir=os.path.join(final_vis_dir,'link_MRCNN')
    
    ##0> load image names  
    imgset_path,jpg_dir,im_ext=io.get_ims_info()   
    im_names=ld.get_im_names(imgset_path)
    im_names=im_names[:test_frm_num]
    # if debug_flag:
    #     print 'imgset_path:',imgset_path
    #     print 'jpg_dir:', jpg_dir
    #     print 'im_ext:', im_ext
    #     print 'im_names:', im_names
    vis_link_imgs(im_names,org_mrcnn_dir,pickd_mrcnn_dir,link_mrcnn_dir)
            
##===================================for testing===============================================

def test_vis_gt_boxes():
    imgset_path,jpg_dir,im_ext=io.get_ims_info()   
    im_names=ld.get_im_names(imgset_path)
    ims=ld.load_color_images(jpg_dir,im_names,im_ext)
    data_dir=os.path.join(mcfg.DATA.DATA_DIR,set_name)   
    tud_dets_gt_path=get_dets_annots_path(data_dir)   
    if debug_flag:
        print 'imgset_path:',imgset_path
        print 'jpg_dir:', jpg_dir
        print 'im_ext:', im_ext
        print 'data_dir:', data_dir
        print 'tud_dets_gt_path:', tud_dets_gt_path
    
    tud_dets_arr=load_dets_annots(tud_dets_gt_path)
   
    frm_ids=tud_dets_arr[:,0].astype(int)
    obj_ids=tud_dets_arr[:,-1].astype(int)
     
    # uniq_obj_ids=np.sort(np.unique(obj_ids))
    
    # for u_id in uniq_obj_ids:
    #     label_row_indexes=np.where(obj_ids==u_id)[0]
    #     label_frm_ids=frm_ids[label_row_indexes]
    #     print 'u_id:',u_id
    #     print 'label_frm_ids:', label_frm_ids
    # if debug_flag:
    #     print 'tud_dets_arr.shape:', tud_dets_arr.shape
    #     print 'frm_ids:', frm_ids
    #     print 'obj_ids:', obj_ids
    #     print 'uniq_obj_ids:', uniq_obj_ids
    coords=tud_dets_arr[:,1:5]
    ##vis_arr=tud_dets_arr[:,5]/100.0   ##make vis range in [0, 1.0]
    boxes=coords_to_boxes(coords)   ## unnormal
    det_dir=mcfg.PROPOSAL.RES_DIR.replace('Continious','Discrete') 
    gt_det_dir=os.path.join(det_dir,'vis_gt_boxes')

    for n_id in xrange(len(im_names)):
        im=ims[n_id]
        im_name=im_names[n_id]
        frm_id=int(im_name.split('_')[1])
        row_indexes=np.where(frm_ids==frm_id)[0]
        frm_boxes=boxes[row_indexes]
        frm_obj_ids=obj_ids[row_indexes]
        vis_gt_bboxes(gt_det_dir,im,im_name,frm_boxes,frm_obj_ids)
    
if __name__ == '__main__':
    ##generate_final_vis_imgs()
    test_vis_gt_boxes()    #tmp