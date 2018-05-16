import _init_paths
from tud_crossing import get_dets_annots_path, get_segs_annots_dir
from tud_crossing import load_dets_annots,load_segs_annots
from box_util import coords_to_boxes
from test_vis import load_frame_masks,load_frame_boxes
from mot_config import mcfg
import io_mot as io
import load_mot as ld
import numpy as np
import os
import cv2

## newest
from tud_crossing import load_dets_new,load_masks_new
from det_util import parse_dets1
from mask_util import masks_align_boxes
from offline_dets import load_dets_ref,load_masks_ref

from eval_util import get_precision_recall, get_average_maskiou

##from track_mask import get_mini_box_from_mask,get_maxbox_mask ##(label_mask,bbox,max_bbox):
debug_flag=True
set_name=mcfg.DATA.SEQNAME
data_dir=os.path.join(mcfg.DATA.DATA_DIR,set_name)
shift_num=mcfg.DATA.SEQ_SHIFT
iter_name='iter0'
im_width=640
im_height=480
over_thre=0.5
refine_flag=True ## final refine mask_rcnn 
init_flag=False  ## init mask_rcnn
cont_flag=False ##

##--------------------------------------common lib ----------------------------------------------------
## get tight box from binary mask
def get_mini_box_from_mask(binarymask):
    one_pos=np.where(binarymask>=1)
    arr_pos=np.asarray(one_pos)

    h_inds=arr_pos[0,:]  #h  ## due to the binarymask(encode in h,w order)
    w_inds=arr_pos[1,:]  #w 

    x_min=min(w_inds)
    x_max=max(w_inds)
    y_min=min(h_inds)
    y_max=max(h_inds)
    w=x_max-x_min+1
    h=y_max-y_min+1
    box=[x_min, y_min, x_max, y_max]
    return box

# get mask with maxbox-size
def get_maxbox_mask(label_mask,bbox,max_bbox):
    #for max_box(through the whole nearby 20f frames)
    x_min=max_bbox[0]
    y_min=max_bbox[1]
    x_max=max_bbox[2]
    y_max=max_bbox[3]
    w_max=x_max-x_min+1
    h_max=y_max-y_min+1

    ##for current image:
    im_x1=max(0,int(np.round(bbox[0])))
    im_y1=max(0,int(np.round(bbox[1])))
    im_w=label_mask.shape[1]
    im_h=label_mask.shape[0]
    im_x2=im_x1+im_w-1
    im_y2=im_y1+im_h-1

    max_label_mask=np.zeros((h_max, w_max), dtype=np.uint8) 
    max_label_mask[im_y1-y_min:im_y2-y_min+1,im_x1-x_min:im_x2-x_min+1]=label_mask
    return max_label_mask

## get binary masks with same size with max_box 
def get_max_masks(masks,boxes,max_box):
    box_num=len(boxes)
    max_masks=[]
    for n_id in xrange(box_num):
        s_mask=masks[n_id]
        box=boxes[n_id]
        max_mask=get_maxbox_mask(s_mask,box,max_box)
        max_masks.append(max_mask)
    return max_masks

##----------------------------------------- gt info ----------------------------------------------------
## get im_names list without .ext
def get_im_names():
    ##0> load image names  
    imgset_path,jpg_dir,im_ext=io.get_ims_info()   
    im_names=ld.get_im_names(imgset_path)
    return im_names

## load gt dets from .txt (tud-crossing): boxes are lose boxes
def get_gt_dets():
    dets_annots_path=get_dets_annots_path(data_dir)
    dets_arr=load_dets_annots(dets_annots_path)
    
    if debug_flag:
        print 'dets_annots_path:', dets_annots_path
        print 'dets_arr.shape:', dets_arr.shape

    frm_ids=dets_arr[:,0].astype(int)
    obj_ids=dets_arr[:,-1].astype(int)
    coords=dets_arr[:,1:5]
    boxes=coords_to_boxes(coords)
    return frm_ids,obj_ids,boxes

## load gt segs from .jpg (tud-crossing),##sort accoring to frm_no
def get_gt_masks():    
    segs_annots_path=get_segs_annots_dir(data_dir)
    segs_arr=load_segs_annots(segs_annots_path)
    return segs_arr

## convert gt_mask from .jpg(in one image) to arrays(several images) 
## from multi-labels to binary labels, and get_tight_boxes
def convert_binary_masks(mask_img,obj_ids):
    #print obj_ids
    obj_num=len(obj_ids)
    if obj_num==0:
        return []
    m_img=mask_img[:,:,0]
    obj_masks=[]
    obj_boxes=[]
    for n_id in xrange(len(obj_ids)):
        tmp_mask=np.zeros((mask_img.shape[0],mask_img.shape[1]))
        obj_id=obj_ids[n_id]        
        m_ids=(m_img==obj_id)
        tmp_mask[m_ids]=1
        # print np.max(tmp_mask)
        # cv2.imshow('tmp_mask',tmp_mask*128)
        # cv2.waitKey(-1)
        mini_box=get_mini_box_from_mask(tmp_mask)
        mini_mask=tmp_mask[mini_box[1]:mini_box[3]+1,mini_box[0]:mini_box[2]+1]
        obj_masks.append(mini_mask)
        obj_boxes.append(mini_box)

    return obj_masks,obj_boxes

def prepare_gt_masks(gt_frm_ids,gt_obj_ids,gt_masks):
    masks=[]
    boxes=[]
    obj_ids=[]
    frm_ids=[]
    uni_ids=np.sort(np.unique(gt_frm_ids))
    # ## calculate based on per-frame
    for n_id in xrange(len(uni_ids)):
        u_id=uni_ids[n_id]
        g_row_indexes=np.where(gt_frm_ids==u_id)[0]
        g_obj_ids=gt_obj_ids[g_row_indexes]      
        ##gt_obj_boxes=gt_boxes[gt_row_indexes] ## org gt_boxes may not that tight when match with gt_masks
        g_mx_mask=gt_masks[n_id]
        g_obj_masks,g_obj_boxes=convert_binary_masks(g_mx_mask,g_obj_ids)
        
        frm_ids.append(u_id)
        obj_ids.append(g_obj_ids)
        boxes.append(g_obj_boxes)
        masks.append(g_obj_masks)
     
    return frm_ids,obj_ids,boxes,masks  

##----------------------------------------- init masks -----------------------------------------------------
def get_init_dets(cont_flag=False):
    det_boxes=[]
    det_masks=[]
    det_obj_ids=[]
    det_frm_ids=[]

    bin_mask_dir=mcfg.MASK.BIN_LAB_DIR 
    init_mask_dir=os.path.join(bin_mask_dir,set_name)
    prop_file_path,prop_file_ext=io.get_dets_info(set_name)
    if not cont_flag:
        init_mask_dir=init_mask_dir.replace('Continious','Discrete')
        prop_file_path=prop_file_path.replace('Continious','Discrete')
    
    prop_arr=ld.load_det_proposals(prop_file_path)
    frm_ids=np.sort(np.unique(prop_arr[:,0]).astype(int))
    frm_num=len(frm_ids)
    img_box=[0,0,im_width-1,im_height-1]   ## not used
         
    ## calculate based on per-frame
    ##for n_id in xrange(1):
    for n_id in xrange(frm_num):
        frm_id=frm_ids[n_id]
        row_indexs=np.where(prop_arr[:,0]==frm_id)[0]      ##frame indexes
        frm_prop_arr=prop_arr[row_indexs]
        frm_labels=frm_prop_arr[:,1].astype(int)
        im_name=set_name+'_'+str(frm_ids[n_id]+shift_num).zfill(6)

        frm_coords=frm_prop_arr[:,2:6]
        flt_frm_boxes=coords_to_boxes(frm_coords)
        frm_masks=load_frame_masks(init_mask_dir,frm_labels,'',im_name)
        int_frm_boxes=boxes_float_to_int(flt_frm_boxes,frm_masks) 
        ##max_masks=get_max_masks(masks,boxes,img_box) ## not used 
        det_frm_ids.append(frm_id)
        det_obj_ids.append(frm_labels)
        det_boxes.append(int_frm_boxes)
        det_masks.append(frm_masks)
    # if debug_flag:
    #     print 'cont_flag:', cont_flag
    #     print 'init_mask_dir:', init_mask_dir
    #     print 'prop_file_path:', prop_file_path
    #     print 'prop_arr.shape:', prop_arr.shape
    #     print 'frm_num:', frm_num
    return det_frm_ids,det_obj_ids,det_boxes,det_masks    

def boxes_float_to_int(flt_boxes,masks):
    box_num=len(flt_boxes)
    int_boxes=np.zeros((box_num,4),int)
    for n_id in xrange(box_num):
        f_box=flt_boxes[n_id]
        box_h,box_w=masks[n_id].shape
        x1=max(0,int(np.round(f_box[0])))
        y1=max(0,int(np.round(f_box[1])))
        x2=x1+box_w-1
        y2=y1+box_h-1
        i_box=[x1,y1,x2,y2]
        int_boxes[n_id]=i_box    
    return int_boxes    
        
##----------------------------------------- picked masks ----------------------------------------------------
def get_refine_dets():
    track_mask_dir=io.get_track_mask_info(set_name)
    prop_file_path,prop_file_ext=io.get_dets_info(set_name)
    prop_arr=ld.load_det_proposals(prop_file_path)
    if debug_flag:
        print 'track_mask_dir:', track_mask_dir
        print 'prop_file_path:', prop_file_path
        print 'prop_file_ext:', prop_file_ext
        print 'prop_arr.shape:', prop_arr.shape
    
    det_frm_ids=[]
    det_obj_ids=[]
    det_boxes=[]
    det_masks=[]
    
    frm_ids=np.sort(np.unique(prop_arr[:,0]).astype(int))
    frm_num=len(frm_ids)

    ## calculate based on per-frame
    for n_id in xrange(frm_num):
        frm_id=frm_ids[n_id]
        row_indexs=np.where(prop_arr[:,0]==frm_id)[0]      ##frame indexes
        frm_prop_arr=prop_arr[row_indexs]
        frm_labels=frm_prop_arr[:,1].astype(int)
        
        im_name=set_name+'_'+str(frm_ids[n_id]+shift_num).zfill(6)   ##im_name
        
        boxes=load_frame_boxes(track_mask_dir,frm_id,frm_labels,iter_name) 
        masks=load_frame_masks(track_mask_dir,frm_labels,iter_name,im_name)        
       
        ##max_masks=get_max_masks(masks,boxes,img_box) ## not used 
        det_frm_ids.append(frm_id)
        det_obj_ids.append(frm_labels)

        det_boxes.append(boxes)
        det_masks.append(masks)

        #if debug_flag:
        #     print 'frm_id:', frm_id
        #     print 'frm_labels:', frm_labels.shape
        #     print 'len(masks):', len(masks)
        #     print 'len(boxes):', len(boxes)
    return det_frm_ids,det_obj_ids,det_boxes,det_masks


# def org_main():    ## not used
    ##--------------------------------------------------------------------------
    ## org(201)
    # # gr_frm_ids,gr_obj_ids,gr_boxes=get_gt_dets() ## n_dets eg:1205
    # # gr_masks=get_gt_masks()   # (n_ims,im_width,im_height) eg=(201,640,480)
    # # gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks=prepare_gt_masks(gr_frm_ids,gr_obj_ids,gr_masks)  ##tight_gt_boxes,    
    # if debug_flag:
    #     print 'gt_masks[0]:', len(gt_masks)    
    #   print 'len(im_names):',im_names
        ##print 'len(gt_frm_ids):,',len(gt_frm_ids),'len(gt_obj_ids):',len(gt_obj_ids), 'len(gt_boxes):',len(gt_boxes)
    # if refine_flag:
        # dt_frm_ids,dt_obj_ids,dt_boxes,dt_masks=get_refine_dets()  # n_frms eg=201
    # if init_flag:
    #     dt_frm_ids,dt_obj_ids,dt_boxes,dt_masks=get_init_dets(cont_flag)  # n_frms eg=199 
    # print len(gt_frm_ids)
        # print len(gt_obj_ids)
        # print len(gt_boxes)
        # print len(gt_masks)

     ##for n_id in xrange():
        # g_obj_ids=gt_obj_ids[n_id]   ## org
        # g_obj_boxes=gt_boxes[n_id]
        # g_obj_masks=gt_masks[n_id]

      # d_obj_ids=dt_obj_ids[n_id]  ## org 
        # d_obj_boxes=dt_boxes[n_id]
        # d_obj_masks=dt_masks[n_id] 


if __name__ == '__main__':
    print '=================tud_crossing_eval===============================' 
    ## input1: gt_dets, gt_masks
    ##im_names=get_im_names()
    gt_dets=load_dets_new()
    gt_masks=load_masks_new()

    gt_frm_ids,gt_obj_ids,gt_boxes=parse_dets1(gt_dets)
    gt_boxes=gt_boxes.astype(int)
    gt_list_masks=masks_align_boxes(gt_masks,gt_boxes)   ## tailor masks    
    gt_masks=np.asarray(gt_list_masks)

    ## input2: dt_dets, dt_masks
    dt_dets=load_dets_ref()
    dt_masks=load_masks_ref()
    dt_frm_ids,dt_obj_ids,dt_boxes=parse_dets1(dt_dets)
         
    dt_boxes=dt_boxes.astype(int)
    dt_list_masks=masks_align_boxes(dt_masks,dt_boxes)   ## tailor masks    
    dt_masks=np.asarray(dt_list_masks)
    
    # if debug_flag:
    #     print 'gt_dets.shape:',gt_dets.shape
    #     print 'gt_masks.shape:', gt_masks[0].shape
    #     print 'gt_boxes:',gt_boxes.shape

    prec,rec=get_precision_recall(gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks,dt_frm_ids,dt_obj_ids,dt_boxes,dt_masks)
    
    ave_maskiou=get_average_maskiou(gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks,dt_frm_ids,dt_obj_ids,dt_boxes,dt_masks)

    if debug_flag:
        print 'prec:', prec
        print 'rec:', rec
        print 'ave_maskiou:', ave_maskiou
  
   

 