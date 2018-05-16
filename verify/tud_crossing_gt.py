## This file is for verifying the new gt_annots of tud-crossing dataset.
## if the new produced gt_masks_tensor(1215,480,640) and new gt_dets(with tight boxes) are correct.
## the tight_boxes are different from the orginal provided bbox.
import _init_paths
import io_mot as io
import load_mot as ld
from tud_crossing import load_dets_new,load_masks_new
from box_util import coords_to_boxes
from vis_util import vis_box_per_person
from common_lib import create_dir,create_recur_dirs
from mot_config import mcfg
import numpy as np
import os

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

## create and get vis_dir
def get_vis_dir():
    res_dir=mcfg.PROPOSAL.RES_DIR
    res_dir=res_dir.replace('Continious','PerPerson')
    create_recur_dirs(res_dir)
    vis_gt_dir=os.path.join(res_dir,'vis_gt_all')
    create_dir(vis_gt_dir)
    return vis_gt_dir

## get gt_masks, gt_boxes, gt_frm_ids, gt_obj_ids...
def parse_gt_annots():
    gt_masks_tensor=load_masks_new()
    gt_dets=load_dets_new()
    ## parse dets
    gt_frm_ids=gt_dets[:,0].astype(int)
    gt_obj_ids=gt_dets[:,1].astype(int)
    gt_coords=gt_dets[:,2:6]
    gt_boxes=coords_to_boxes(gt_coords)  ## tight_gt_boxes
    return gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks_tensor

##---------------------------------------  vis bridge ---------------------------------------------------
## vis new gt_annots(boxes and masks) per person
def vis_gt_per_person(gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks,gt_boxes1=[]):
    uniq_obj_ids=np.sort(np.unique(gt_obj_ids))
    for obj_id in uniq_obj_ids:
        row_indexes=np.where(gt_obj_ids==obj_id)[0]        
        label_frm_ids=gt_frm_ids[row_indexes]
        label_boxes=gt_boxes[row_indexes]   ## tight box
        if len(gt_boxes1)>0:
            label_boxes1=gt_boxes1[row_indexes]  ## loose box
        else:
            label_boxes1=[]
        label_masks=gt_masks[row_indexes]
        id_indexes=label_frm_ids-1
        label_ims=rgb_ims[id_indexes]
        label_im_names=im_names[id_indexes]
        label_dir=os.path.join(vis_gt_dir,str(obj_id))
        create_dir(label_dir)
        vis_box_per_person(label_dir,label_ims,label_im_names,label_boxes,obj_id,label_masks,label_boxes1)


if __name__ == '__main__':
    print '=================verify tud_crossing_gt===================='
    im_names=get_im_names()
    rgb_ims=load_color_ims(im_names)
    vis_gt_dir=get_vis_dir()
    gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks_tensor=parse_gt_annots()

    if debug_flag:
        print 'vis_gt_dir:', vis_gt_dir
        print gt_masks_tensor.shape
        print gt_boxes.shape
        
    vis_gt_per_person(gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks_tensor)
   
    