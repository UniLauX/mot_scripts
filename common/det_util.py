import _init_paths
import numpy as np
from box_util import coords_to_boxes,boxes_to_coords
from mot_imgnet.mot_util import write_detarr_as_mot

## filter dets based on given frm_ids
def dets_filter_frms(in_dets,frm_ids):
    ##print '============dets_filter_frms=================='
    row_indexes=np.empty((0,),dtype=int)
    if len(frm_ids)==0:
        out_dets=in_dets
        row_indexes=np.arange(in_dets)
    else:
        for frm_id in frm_ids:
            r_indexes=np.where(in_dets[:,0]==frm_id)[0]
            row_indexes=np.hstack((row_indexes,r_indexes))
        out_dets=in_dets[row_indexes,:]
    return out_dets,row_indexes

## filter dets based on given obj_ids
def dets_filter_ids(in_dets,obj_ids):
    ##print '==============dets_filter_ids================='
    row_indexes=np.empty((0,),dtype=int)
    if len(obj_ids)==0:
        out_dets=in_dets
        row_indexes=np.arange(in_dets)
    else:
        for obj_id in obj_ids:
            r_indexes=np.where(in_dets[:,1]==obj_id)[0]
            row_indexes=np.hstack((row_indexes,r_indexes))
        out_dets=in_dets[row_indexes,:]
        #print 'out_dets.shape:', out_dets.shape
    return out_dets,row_indexes

## filter dets based on given score_thereshod
def dets_filter_scores(in_dets,score_thre):
    print '===============dets_filter_scores============='    

## get gt_frm_ids, gt_obj_ids,gt_boxes,...(may need to add gt_scores,gt_vis,gt_ious etc)
def parse_dets1(dets):
    ## parse dets
    frm_ids=dets[:,0].astype(int)
    obj_ids=dets[:,1].astype(int)
    coords=dets[:,2:6]
    boxes=coords_to_boxes(coords)  ## tight_gt_boxes
    return frm_ids,obj_ids,boxes

## form dets(arr) from frm_ids, obj_ids, boxes
def form_dets1(frm_ids,obj_ids,boxes):
    coords=boxes_to_coords(boxes)
    row_num=len(frm_ids)
    dets=np.zeros((row_num,6))
    dets[:,0]=frm_ids
    dets[:,1]=obj_ids
    dets[:,2:6]=coords
    return dets

## save dets_arr(np array) into local disk(.txt)
def save_dets(dets_path,dets_arr):
    write_detarr_as_mot(dets_path,dets_arr)













    








