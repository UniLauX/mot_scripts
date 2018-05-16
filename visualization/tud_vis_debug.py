import _init_paths
import io_mot as io
import load_mot as ld
from tud_crossing import get_gt_dets, get_gt_masks,prepare_gt_masks
from vis_util import vis_box_per_person
from box_util import coords_to_boxes
from common_lib import create_dir,create_recur_dirs,get_mini_box_from_mask
from mot_config import mcfg
import os
import numpy as np
import cv2

per_flag=True
tight_box_flag=True ## if flase, tight_gt_boxes=gt_boxes
vis_gt_flag=True
## vis org. interplate, picked
debug_flag=True
set_name=mcfg.DATA.SEQNAME
data_dir=os.path.join(mcfg.DATA.DATA_DIR,set_name)

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

## create and get vis_dir
def get_vis_dir():
    res_dir=mcfg.PROPOSAL.RES_DIR
    if per_flag:
        res_dir=res_dir.replace('Continious','PerPerson')
        create_recur_dirs(res_dir)
    vis_gt_dir=os.path.join(res_dir,'vis_gt_all')
    create_dir(vis_gt_dir)
    return vis_gt_dir

def get_tud_gt_annots():

    gt_frm_ids,gt_obj_ids,gt_boxes=get_gt_dets() ## n_dets eg:1215
    gr_masks=get_gt_masks()   # (n_ims,im_width,im_height) eg=(201,640,480)
    tight_gt_boxes,gt_masks=prepare_gt_masks(gt_frm_ids,gt_obj_ids,gr_masks,gt_boxes,tight_box_flag) 
    return gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks,tight_gt_boxes

def vis_gt_per_person(gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks,tight_gt_boxes=[]):
    uniq_obj_ids=np.sort(np.unique(gt_obj_ids))
    for obj_id in uniq_obj_ids:
        row_indexes=np.where(gt_obj_ids==obj_id)[0]        
        label_frm_ids=gt_frm_ids[row_indexes]
        label_boxes=tight_gt_boxes[row_indexes]   ## tight box
        label_boxes1=gt_boxes[row_indexes]  ## loose box

        label_masks=gt_masks[row_indexes]
        id_indexes=label_frm_ids-1
        label_ims=rgb_ims[id_indexes]
        label_im_names=im_names[id_indexes]
        label_dir=os.path.join(vis_gt_dir,str(obj_id))
        create_dir(label_dir)
        vis_box_per_person(label_dir,label_ims,label_im_names,label_boxes,obj_id,label_masks,label_boxes1)
        
##(1) vis gt_seg and gt_box
if __name__ == '__main__':
    print '=================tud_vis_debug===============================' 
    im_names=get_im_names()
    rgb_ims=load_color_ims(im_names)
    vis_gt_dir=get_vis_dir()

    gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks,tight_gt_boxes=get_tud_gt_annots()
  
  
    if vis_gt_flag:
        vis_gt_per_person(gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks,tight_gt_boxes)
  
    # if debug_flag:
    #     print gt_frm_ids.shape
    #     print gt_obj_ids.shape
    #     print gt_boxes.shape
    #     print gt_masks.shape
    #     print vis_gt_dir
    
    
  
        






