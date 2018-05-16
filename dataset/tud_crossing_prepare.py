from tud_crossing import get_org_gt_dets, get_org_gt_masks,prepare_binary_gt_masks
from tud_crossing import save_masks_new, save_dets_new,load_masks_new,load_dets_new
from tud_crossing import save_dets_org, load_dets_org
from det_util import form_dets1
import numpy as np
tight_box_flag=True
debug_flag=True

##------------------------------prepare gt_annots--------------------------------------------
## 1> prepare gt_dets with lose_gt_boxes(org provided)
def prepare_lose_gt_dets():
    gt_frm_ids,gt_obj_ids,gt_boxes=get_org_gt_dets() ## n_dets eg:1215
    dets=form_dets1(gt_frm_ids,gt_obj_ids,gt_boxes)
    save_dets_org(dets)

## make all the binary mask a tensor, shape=(1215,480,640)
## write new dets based on tight_boxes(get from masks)
def prepare_gt_masks():
    print '=================prepare gt masks===============================' 
    gt_frm_ids,gt_obj_ids,gt_boxes=get_org_gt_dets() ## n_dets eg:1215  ##note:lose gt_boxes
    gr_masks=get_org_gt_masks()   # (n_ims,im_width,im_height) eg=(201,640,480)
    
    if debug_flag:
        print 'len(gr_masks):',len(gr_masks)
    
    ##tight_gt_boxes,gt_masks=prepare_binary_gt_masks(gt_frm_ids,gt_obj_ids,gr_masks,gt_boxes,tight_box_flag) 
    
    '''
    gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks,tight_gt_boxes=get_tud_gt_annots()
    save_masks_new(gt_masks)
    '''
# def get_tud_gt_annots():
#     gt_frm_ids,gt_obj_ids,gt_boxes=get_gt_dets() ## n_dets eg:1215
#     gr_masks=get_gt_masks()   # (n_ims,im_width,im_height) eg=(201,640,480)
#     tight_gt_boxes,gt_masks=prepare_gt_masks(gt_frm_ids,gt_obj_ids,gr_masks,gt_boxes,tight_box_flag) 
#     return gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks,tight_gt_boxes


# ## prepare gt_dets with tight_gt_boxes
# def prepare_tight_gt_dets():

#     gt_dets=form_dets1(gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks,tight_gt_boxes)
#     save_dets_new(gt_dets)


##*** finish this function later....

##(1) vis gt_seg and gt_box
if __name__ == '__main__':
    prepare_lose_gt_dets()
    prepare_gt_masks()
    
    '''
    gt_masks_tensor=load_masks_new()
    gt_dets=load_dets_new()
    print gt_masks_tensor.shape
    print gt_dets.shape
    '''
    ## 1>prepare gt_masks
    ## 2>prepare tight_gt_boxes
    ## 3>prepare lose_gt_boxes

    