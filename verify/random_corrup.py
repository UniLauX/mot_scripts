## The purpose of this file is to verify the 'pick_mask' algorithm.
## in specific, using the gt_masks as input. Random corrup(erosion or dilation) them
## according to the result to verity the 'pick_mask''s performance(both disadvantages and advantages)
import _init_paths
import io_mot as io
import load_mot as ld
from tud_crossing import load_dets_new,load_masks_new
from det_util import dets_filter_frms,dets_filter_ids,parse_dets1
from image_util import save_images
from vis_mask import vis_alpha_mask, vis_im_mask
from common_lib import create_dir
import numpy as np
import cv2
from mot_config import mcfg
import os
from mask_util import get_minbox_mask,masks_align_boxes
from track_mask import refine_masks
from box_util import get_mini_box_from_mask,boxes_to_coords,coords_to_boxes

from eval_util import get_precision_recall, get_average_maskiou
from offline_dets import get_ref_dets,get_ref_masks,get_enlarge_masks

from eval_util import obj_ids_map

seq_name=mcfg.DATA.SEQNAME
im_ext=mcfg.DATA.IMGEXT
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

## filter dets
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

## random corrup single mask
def random_corrup_mask(in_mask):
    out_mask=in_mask.copy()

    kernel_bound=10
    iter_bound=5
    type_num=3   ## 0: erosion  1: dilation  2: iter1

    ##corrup_type_list=['erosion','dilation']
    ##=np.random.choice(mask_num,corrup_num,replace=False)
    kernal_size=np.random.randint(1,kernel_bound)
    iter_times=np.random.randint(1,iter_bound)
    corrup_type=np.random.randint(0,type_num)

    # if debug_flag:
    #     print 'kernal_size:', kernal_size
    #     print 'iter_times:', iter_times
    #     print 'corrup_type:', corrup_type

    kernel = np.ones((kernal_size,kernal_size), np.uint8)
    if corrup_type==0:
            out_mask = cv2.erode(in_mask, kernel, iterations=iter_times)
    if corrup_type==1:
            out_mask = cv2.dilate(in_mask, kernel, iterations=iter_times)
    if corrup_type==2:
            out_mask=random_corrup_mask(in_mask)
    
    ## avoid empty mask
    while np.max(out_mask) != 1:
        out_mask=random_corrup_mask(in_mask)

    return out_mask

## corrup masks uing 
def get_corrup_masks(in_masks,corrup_ratio=0.2,sample_type='average'):

    ## set corrup ratio and decide corrup frames
    out_masks=np.zeros(in_masks.shape,dtype=np.uint8)
    out_masks=np.copy(in_masks)  # value copy

    ## 1> first, according to the corrup_ratio to get the corrup_indexes(finished)
    mask_num=len(in_masks)
    corrup_num=min(mask_num,int(np.round(mask_num*corrup_ratio)))
    corrup_indexes=[]
    if sample_type=='random':
        corrup_indexes=np.sort(np.random.choice(mask_num,corrup_num,replace=False))  # random sampling

    if sample_type=='average':                    ## average sampling            
        corrup_inter=mask_num/(corrup_num+0.0)
        for c_id in xrange(corrup_num):   
            corrup_id=corrup_inter*c_id  
            select_id=min(mask_num-1, int(np.round(corrup_id))) 
            corrup_indexes+=[select_id]
        corrup_indexes=np.array(corrup_indexes)

    # print 'corrup_indexes:', corrup_indexes
    corrup_masks=out_masks[corrup_indexes]
    for n_id in xrange(corrup_num):
        i_mask=corrup_masks[n_id]
        o_mask=random_corrup_mask(i_mask)
        corrup_masks[n_id]=o_mask

    out_masks[corrup_indexes]=corrup_masks
    return out_masks

##  set up all the parameters in this file
def parse_args():
    # 0> parameters
    start_frm_id=1
    s_frm=1
    e_frm=201
    s_index=s_frm-start_frm_id
    e_index=e_frm-start_frm_id
    frm_ids=np.arange(s_frm,e_frm+1)

    obj_ids=np.array([14])
    ##obj_ids=np.array([5, 10, 12, 13, 14])
    return frm_ids, obj_ids, start_frm_id

### create gt_masks_dir, corrup_masks_dir and pick_masks_dir for later use
def create_masks_dirs(obj_id):
    ##print '=========================get maks dir====================='
    # seg_dir
    seg_dir=mcfg.EXPR.SEG_DIR 
    dataset=mcfg.DATA.DATASET
    seg_algr=mcfg.EXPR.SEG_ALGR
    seg_dir=os.path.join(seg_dir,dataset,seg_algr)

    # random_corrup_dir
    rand_corrup_dir=os.path.join(seg_dir,'random_corrup')
    create_dir(rand_corrup_dir)

    # sub masks dirs
    gt_masks_dir=os.path.join(rand_corrup_dir,'gt_masks')
    corrup_masks_dir=os.path.join(rand_corrup_dir,'corrup_masks')
    pick_masks_dir=os.path.join(rand_corrup_dir,'pick_masks')
    create_dir(gt_masks_dir)
    create_dir(corrup_masks_dir)
    create_dir(pick_masks_dir)
    
    ## sub masks dirs with obj_ids
    gt_masks_dir=os.path.join(gt_masks_dir,str(obj_id))
    create_dir(gt_masks_dir)

    corrup_masks_dir=os.path.join(corrup_masks_dir,str(obj_id))
    create_dir(corrup_masks_dir)

    # pick_masks_dir=os.path.join(pick_masks_dir,str(obj_id))
    # create_dir(pick_masks_dir)
    return gt_masks_dir,corrup_masks_dir,pick_masks_dir

## save visualized gt_masks and corrupped masks
def save_vis_masks(ims,im_names,gt_masks_dir,gt_masks,corrup_masks_dir,corrup_masks):
    # 5.2> save vis_gt_masks and corrup_masks 
    alpha_folder='alpha_masks'
    color_folder='color_masks'

    ## alpha masks(gt)
    alpha_gt_dir=os.path.join(gt_masks_dir,alpha_folder)
    create_dir(alpha_gt_dir)
    alpha_gt_masks=gt_masks*255.0
    save_images(alpha_gt_masks,alpha_gt_dir,im_names,im_ext)
   
    ## alpha masks(corrup)
    alpha_corrup_dir=os.path.join(corrup_masks_dir,alpha_folder) 
    create_dir(alpha_corrup_dir)
    alpha_corrup_masks=corrup_masks*255.0
    save_images(alpha_corrup_masks,alpha_corrup_dir,im_names,im_ext)
     
    ## color masks(gt)
    color_gt_masks=np.zeros((ims.shape),dtype=np.uint8)     
    color=[0,0,255]
    for m_id in xrange(len(gt_masks)):
        im=ims[m_id]
        mask=gt_masks[m_id]
        color_mask=vis_im_mask(im,mask,color,True)
        color_gt_masks[m_id]=color_mask
    
    color_gt_dir=os.path.join(gt_masks_dir,color_folder)
    create_dir(color_gt_dir)
    save_images(color_gt_masks,color_gt_dir,im_names,im_ext)
    
    ## here is small bug(need to debug)
    ## color masks(corrup)
    color_corrup_masks=np.zeros((ims.shape),dtype=np.uint8)     
    color=[0,0,255]
    for m_id in xrange(len(corrup_masks)):
        im=ims[m_id]
        mask=corrup_masks[m_id]
        color_mask=vis_im_mask(im,mask,color,True)
        color_corrup_masks[m_id]=color_mask
    
    color_corrup_dir=os.path.join(corrup_masks_dir,color_folder)
    create_dir(color_corrup_dir)
    save_images(color_corrup_masks,color_corrup_dir,im_names,im_ext)


## get boxes(coords) that ecncompass the masks's shape
def boxes_encompass_masks(masks):
    mask_num=len(masks)
    boxes=np.zeros((mask_num,4),dtype=int)
    for n_id in xrange(mask_num):
        mask=masks[n_id]
        box=get_mini_box_from_mask(mask)
        boxes[n_id,:]=box
    return boxes


## save verify results in local disk
def save_verify_res(file_path,res_content):
    print 'save verify results in local disk...'
    print 'res_content:', res_content

    with open(file_path, 'w') as f:     
            f.write('{:s}\n'. format(res_content))
    f.close()  
    

## m1: using metric numbet to verify gt annots
def verify_gt_annots(gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks,write_res_flag=False):
    print '===========================m1.verify gt annots============================'
    ## switch 1: for gt
    # ## 5> postprocessing masks
    gt_boxes=gt_boxes.astype(int)
    gt_list_masks=masks_align_boxes(gt_masks,gt_boxes)   ## tailor masks    
    gt_masks=np.asarray(gt_list_masks)
    # ## 6> verify the gt_annots and measurement algorithm 
    prec,rec,tp_cnt,fp_cnt,miss_gt_cnt=get_precision_recall(gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks,gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks) 
    ave_mask_iou_arr,ave_inst_iou=get_average_maskiou(gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks,gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks)

    if debug_flag:
        print 'tp_cnt:', tp_cnt
        print 'fp_cnt:', fp_cnt
        print 'miss_gt_cnt:', miss_gt_cnt
        print 'prec:', prec
        print 'rec:', rec
        print 'ave_mask_iou_arr:', ave_mask_iou_arr
        print 'ave_inst_iou:', ave_inst_iou

    ## write metrics in local disk
    if write_res_flag:
        metric_dir=get_metric_dir()
        gt_metric_dir=os.path.join(metric_dir,'gt.txt')   
        res_content='-------------------------gt----------------------------'
        res_content+='\n'+'obj_ids:'+str(np.sort(np.unique(gt_obj_ids)))
        res_content+='\n'+'tp_cnt:'+str(tp_cnt)
        res_content+='\n'+'fp_cnt:'+str(fp_cnt)
        res_content+='\n'+'miss_gt_cnt:'+str(miss_gt_cnt)
        res_content+='\n'+'prec:'+str(prec)
        res_content+='\n'+'rec:'+str(rec)
        res_content+='\n'+'ave_mask_iou_arr:'+str(ave_mask_iou_arr)
        res_content+='\n'+'ave_inst_iou:'+str(ave_inst_iou)
        save_verify_res(gt_metric_dir,res_content) 

## m2: using metric numbet to verify corrup annots
def verify_corrup_annots(gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks,corrup_boxes,corrup_masks,write_res_flag=False):
    print '===========================m2.verify corrup annots============================'
    ## switch2: for corrup one
    #  preprocessing gt annots
    gt_boxes=gt_boxes.astype(int)
    gt_list_masks=masks_align_boxes(gt_masks,gt_boxes)   ## tailor masks    
    gt_masks=np.asarray(gt_list_masks)
    
    # preprocessing corrup annots
    corrup_boxes=corrup_boxes.astype(int) 
    tailor_masks=masks_align_boxes(corrup_masks,corrup_boxes)
    tailor_masks=np.asarray(tailor_masks)

    # verify the corrup_annots and measurement algorithm 
    prec,rec,tp_cnt,fp_cnt,miss_gt_cnt=get_precision_recall(gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks,gt_frm_ids,gt_obj_ids,corrup_boxes,tailor_masks) 
    ave_mask_iou_arr,ave_inst_iou=get_average_maskiou(gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks,gt_frm_ids,gt_obj_ids,corrup_boxes,tailor_masks)
    if debug_flag:
        print 'prec:', prec
        print 'rec:', rec
        print 'ave_mask_iou_arr:', ave_mask_iou_arr
        print 'ave_inst_iou:', ave_inst_iou

    ## write metrics in local disk
    if write_res_flag:
        metric_dir=get_metric_dir()
        gt_metric_dir=os.path.join(metric_dir,'corrup.txt')   
        res_content='-------------------------corrup----------------------------'
        res_content+='\n'+'obj_ids:'+str(np.sort(np.unique(gt_obj_ids)))
        res_content+='\n'+'tp_cnt:'+str(tp_cnt)
        res_content+='\n'+'fp_cnt:'+str(fp_cnt)
        res_content+='\n'+'miss_gt_cnt:'+str(miss_gt_cnt)
        res_content+='\n'+'prec:'+str(prec)
        res_content+='\n'+'rec:'+str(rec)
        res_content+='\n'+'ave_mask_iou_arr:'+str(ave_mask_iou_arr)
        res_content+='\n'+'ave_inst_iou:'+str(ave_inst_iou)
        save_verify_res(gt_metric_dir,res_content)         


## m3: using metric numbet to verify ref annots
def verify_ref_annots(gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks,ref_boxes,ref_masks,write_res_flag):
    print '===========================m3.verify refine annots============================'
    ## switch3: for refine one
    #  preprocessing gt annots
    gt_boxes=gt_boxes.astype(int)
    gt_list_masks=masks_align_boxes(gt_masks,gt_boxes)   ## tailor masks    
    gt_masks=np.asarray(gt_list_masks)
    
    # preprocessing corrup annots
    ref_boxes=ref_boxes.astype(int) 
    tailor_masks=masks_align_boxes(ref_masks,ref_boxes)
    tailor_masks=np.asarray(tailor_masks)

    # verify the corrup_annots and measurement algorithm 
    prec,rec,tp_cnt,fp_cnt,miss_gt_cnt=get_precision_recall(gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks,gt_frm_ids,gt_obj_ids,ref_boxes,tailor_masks) 
    ave_mask_iou_arr,ave_inst_iou=get_average_maskiou(gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks,gt_frm_ids,gt_obj_ids,ref_boxes,tailor_masks)
    
    if debug_flag:
        print 'prec:', prec
        print 'rec:', rec
        print 'ave_mask_iou_arr:', ave_mask_iou_arr
        print 'ave_inst_iou:', ave_inst_iou

    ## write metrics in local disk
    if write_res_flag:
        metric_dir=get_metric_dir()
        gt_metric_dir=os.path.join(metric_dir,'refine.txt')   
        res_content='-------------------------refine----------------------------'
        res_content+='\n'+'obj_ids:'+str(np.sort(np.unique(gt_obj_ids)))
        res_content+='\n'+'tp_cnt:'+str(tp_cnt)
        res_content+='\n'+'fp_cnt:'+str(fp_cnt)
        res_content+='\n'+'miss_gt_cnt:'+str(miss_gt_cnt)
        res_content+='\n'+'prec:'+str(prec)
        res_content+='\n'+'rec:'+str(rec)
        res_content+='\n'+'ave_mask_iou_arr:'+str(ave_mask_iou_arr)
        res_content+='\n'+'ave_inst_iou:'+str(ave_inst_iou)
        save_verify_res(gt_metric_dir,res_content)               

# def test_corrup_mask():
    # gt_masks_tensor=load_masks_new()
    # in_mask=gt_masks_tensor[5]
    # out_mask=random_corrup_mask(in_mask)

    # alpha_in_mask=in_mask*255.0 
    # cv2.imshow('in_mask',alpha_in_mask)

    # alpha_out_mask=out_mask*255.0
    # cv2.imshow('out_mask',alpha_out_mask)
    # cv2.waitKey(-1)

    # if debug_flag:
    #     print 'in_mask.shape:', in_mask.shape
    
    ### note: at the moment, just 1 instance allowed every time.
    ## methods for corrupping masks
    ##1.introducing noise in the boundary 
    ##2.truncating the mask
    ##3.dilation and erosion
    ##4.crop masks
    ##5.jitter the box position 
def cal_per_label(rgb_ims,im_names,forward_flows,backward_flows,filt_obj_ids,gt_dets,gt_masks,corrup_boxes,corrup_masks):
    print '===============calculate per label================='
    ## parameters
    start_frm_id=1 

    for f_obj_id in filt_obj_ids:
        print '---------------------------------------------',f_obj_id
        gt_masks_dir,corrup_masks_dir,pick_masks_dir=create_masks_dirs(f_obj_id)
        
        o_gt_dets,row_indexes=dets_filter_ids(gt_dets,[f_obj_id])

        o_gt_frm_ids,o_gt_obj_ids,o_gt_boxes=parse_dets1(o_gt_dets) 
        
        o_gt_masks=gt_masks[row_indexes]

        o_corrup_boxes=corrup_boxes[row_indexes]

        o_corrup_masks=corrup_masks[row_indexes]

        ## filter color images
        filt_indexes=o_gt_frm_ids-start_frm_id

        im_arr=np.asarray(rgb_ims)
        cur_ims=im_arr[filt_indexes]
        cur_im_names=im_names[filt_indexes]
        fw_flows=forward_flows[filt_indexes[:-1]]
        bw_flows=backward_flows[filt_indexes[:-1]]

        save_vis_masks(cur_ims,cur_im_names,gt_masks_dir,o_gt_masks,corrup_masks_dir,o_corrup_masks)  #v1

        if debug_flag:
            print 'o_gt_dets:', o_gt_dets
            print 'gt_masks_dir:',gt_masks_dir
            print 'corrup_masks_dir:', corrup_masks_dir
            print 'pick_masks_dir:', pick_masks_dir
            print 'o_gt_frm_ids:', o_gt_frm_ids

        #----------------------------------------------------------------------------------------
        ## prepare dataset for 'refine-masks' algorithm
        # preprocessing corrup_masks and corrup_boxes(corrup_dets): make masks has the same size with boxes
        o_corrup_boxes=o_corrup_boxes.astype(int) 
        tailor_masks=masks_align_boxes(o_corrup_masks,o_corrup_boxes)
        tailor_masks=np.asarray(tailor_masks)

        ## prepare (corrup) dets
        corrup_coords=boxes_to_coords(o_corrup_boxes)
        corrup_dets=o_gt_dets
        corrup_dets[:,2:6]=corrup_coords
        
        # 6> refine masks
        ##input: prop_arr,masks,obj_ids, forward_flows, backward_flows
        refine_masks(pick_masks_dir,rgb_ims,forward_flows,backward_flows,tailor_masks,corrup_dets)
        
        ref_dets=get_ref_dets(pick_masks_dir,corrup_dets)   ## verified
        # ref_masks=get_ref_masks(pick_masks_dir,corrup_dets)

        ref_masks=get_enlarge_masks(pick_masks_dir,corrup_dets)
        ref_coords=ref_dets[:,2:6]
        ref_boxes=coords_to_boxes(ref_coords)

        verify_ref_annots(gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks,ref_boxes,ref_masks,write_res_flag)

        # if debug_flag:
        #     print 'ref_dets.shape:', ref_dets.shape
        #     ##print 'ref_dets:', ref_dets
        #     print 'ref_masks.shape:',ref_masks.shape


##==========================================metric numbers============================================================
def get_metric_dir():
    metric_dir='/media/uni/uni1/pickmask/tud-crossing/random_corrup/verify_metrics' 
    return metric_dir

##note: lp_solve problem may caused by empty mask.
## posible solution: if empty, reproduce it again.
if __name__ == '__main__':
    print '================verify random corrup======================'
    ## test random sampling...
    ## parameters
    corrup_ratio=0.8
    sample_type='random'
    ##sample_type='average'
    write_res_flag=True
    
    # 0> parameters and filters
    filt_frm_ids, filt_obj_ids, start_frm_id=parse_args()
    frm_indexes=filt_frm_ids-start_frm_id

    # 1> im_names and ims
    im_names=get_im_names()          #whole
    im_names=im_names[frm_indexes]   
    rgb_ims=load_color_ims(im_names) 

    # 2> load optical flows
    optflow_dir,flow_im_ext=io.get_flows_info(seq_name)
    forward_flows,backward_flows=ld.load_opticalflows_flo(optflow_dir,im_names,flow_im_ext)
  
     # 3> gt masks and dets
    gt_dets,gt_masks=get_filter_gt_annots(filt_frm_ids,filt_obj_ids)
    gt_frm_ids,gt_obj_ids,gt_boxes=parse_dets1(gt_dets) 
    
    verify_gt_annots(gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks,write_res_flag) ## M1
    
    # if debug_flag:
    #     print 'filt_frm_ids:', filt_frm_ids
    #     print 'filt_obj_ids:', filt_obj_ids
    #     print 'frm_indexes:', frm_indexes
    #     print 'len(im_names)', (im_names)
    #     print 'len(ims):', len(rgb_ims)
    #     print 'len(forward_flows):',len(forward_flows)
    #     print 'gt_boxes.shape:', gt_boxes.shape
    #     print 'gt_masks.shape:', gt_masks.shape

    ##---------------------------------------------------------------------------------
    # if debug_flag:
    #     print 'filt_obj_ids:',filt_obj_ids
    #     print 'frm_indexes:', frm_indexes
    
    ## 4> get corrup masks and corrup boxes
    corrup_masks=get_corrup_masks(gt_masks,corrup_ratio,sample_type)    

    corrup_boxes=boxes_encompass_masks(corrup_masks)
    # if debug_flag:
    #     print rgb_ims.shape
    #     print gt_masks.shape
    #     print corrup_masks.shape
    verify_corrup_annots(gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks,corrup_boxes,corrup_masks,write_res_flag) ## M2    
 
    ## add multi-instances 
    cal_per_label(rgb_ims,im_names,forward_flows,backward_flows,filt_obj_ids,gt_dets,gt_masks,corrup_boxes,corrup_masks)
    ## > save_vis_results
    # 5> create masks dirs
    
    
    
  
