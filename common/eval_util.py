## This function aims to get metric numbers.
## at moment,it uses gt_masks and gt_dets to align, thus no false postive 
import numpy as np

debug_flag=True

## better to check later...
## box1 and mask1 have same size. box2 and mask2 as well
def mask_overlap(box1, box2, mask1, mask2):
    """
    box1: gtmask bound  
    box2: predicted mask bound
    mask1: gt mask (contained in gtmask bound)
    mask2: predicted mask (only in predicted box bound)
    
    This function calculate region IOU when masks are
    inside different boxes
    Returns:
        intersection over unions of this two masks
    """
    #overlap
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x1 > x2 or y1 > y2:
        return 0
    # width, height of overlap box
    w = x2 - x1 + 1
    h = y2 - y1 + 1
  
    # print 'mask_sum:',mask1.sum(), mask2.sum()
    # print 'w:',w, 'h:',h
    # get masks in the intersection part
    start_ya = y1 - box1[1]
    start_xa = x1 - box1[0]
    inter_maska = mask1[start_ya: start_ya + h, start_xa:start_xa + w] #this is gt mask of the overlap box

    start_yb = y1 - box2[1]
    start_xb = x1 - box2[0]
    inter_maskb = mask2[start_yb: start_yb + h, start_xb:start_xb + w]

    assert inter_maska.shape == inter_maskb.shape
    # cv2.imshow('mask1:', inter_maska*128)
    # cv2.imshow('mask2:',inter_maskb*128)
    # cv2.waitKey(-1)
    inter = np.logical_and(inter_maskb, inter_maska).sum()
    union = mask1.sum() + mask2.sum() - inter
    if union < 1.0:
        return 0
    return float(inter) / float(union)


## get precision and recall
def get_precision_recall(gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks,dt_frm_ids,dt_obj_ids,dt_boxes,dt_masks,over_thre=0.5):
    # print 'calculate precision recall...'
    prec=0.0
    rec=0.0

    uniq_frm_ids=np.sort(np.unique(dt_frm_ids))
    frm_num=len(uniq_frm_ids)

    npos=len(gt_boxes)    ## gt_num
    tp_cnt=0 ## true positive 
    fp_cnt=0 ## false positive
    miss_gt_cnt=0 ## due to occulution, some objects miss gt_annots even though they are in the middle of the sequence.    
    ## calculate based on per-frame
    for n_id in xrange(frm_num):
        frm_id=uniq_frm_ids[n_id]

        frm_indexes1=np.where(gt_frm_ids==frm_id)[0]
        g_obj_ids=gt_obj_ids[frm_indexes1]
        g_obj_boxes=gt_boxes[frm_indexes1]
        g_obj_masks=gt_masks[frm_indexes1]

        frm_indexes2=np.where(dt_frm_ids==frm_id)[0]
        d_obj_ids=dt_obj_ids[frm_indexes2]
        d_obj_boxes=dt_boxes[frm_indexes2]
        d_obj_masks=dt_masks[frm_indexes2]

        for k_id in xrange(len(d_obj_ids)):
            d_obj_id=d_obj_ids[k_id]
            #f_index=np.where(g_obj_ids==d_obj_id)[0]
            f_index=np.where(g_obj_ids==d_obj_id)[0]
            if len(f_index)==0:
                miss_gt_cnt=miss_gt_cnt+1
            else:
                # box1=g_obj_boxes[int(f_index)]          
                # mask1=g_obj_masks[int(f_index)]
                f_index=int(f_index)
                box1=g_obj_boxes[f_index]          
                mask1=g_obj_masks[f_index]
                box2=d_obj_boxes[k_id]
                ##box2=d_obj_boxes[k_id].astype(int)
                mask2=d_obj_masks[k_id]

                mask_over=mask_overlap(box1,box2,mask1,mask2)

                if mask_over>=over_thre:
                    tp_cnt=tp_cnt+1
                else:
                    fp_cnt=fp_cnt+1
    rec = tp_cnt / float(npos)
    prec= tp_cnt / np.maximum(tp_cnt + fp_cnt, np.finfo(np.float64).eps)
    
    # print 'tp_cnt:', tp_cnt
    # print 'fp_cnt:', fp_cnt
    # print 'miss_gt_cnt:', miss_gt_cnt
    return prec,rec,tp_cnt,fp_cnt,miss_gt_cnt 

## map obj_ids to obj_indexes
def obj_ids_map(obj_ids):
    uniq_obj_ids=np.sort(np.unique(obj_ids))
    obj_num=len(uniq_obj_ids)
    map_indexes=np.arange(obj_num)
    map_list={}
    
    for n_id in xrange(obj_num):
        obj_id=uniq_obj_ids[n_id]
        map_list[obj_id]= n_id

    return map_list


## get average maskiou per-object and (total) average maskiou
def get_average_maskiou(gt_frm_ids,gt_obj_ids,gt_boxes,gt_masks,dt_frm_ids,dt_obj_ids,dt_boxes,dt_masks):
    #print '========================average maskiou============================================'    
    obj_num=len(np.unique(gt_obj_ids))
    mask_iou_arr=np.zeros(obj_num,dtype=float)
    mask_iou_cnt=np.zeros(obj_num,dtype=int)
    ave_mask_iou_arr=np.zeros(obj_num,dtype=float)
    ave_inst_iou=0.0     ## average maskiou of all instances
    miss_gt_cnt=0

    obj_ids_map_list=obj_ids_map(gt_obj_ids)  # map the actual obj_ids to indexes start from 0
    #     print 'gt_masks.shape:',gt_masks.shape
    #     print 'dt_masks.shape:', dt_masks.shape
    uniq_frm_ids=np.sort(np.unique(dt_frm_ids))
    frm_num=len(uniq_frm_ids)

    ## calculate based on per-frame
    for n_id in xrange(frm_num):
        frm_id=uniq_frm_ids[n_id]

        frm_indexes1=np.where(gt_frm_ids==frm_id)[0]
        g_obj_ids=gt_obj_ids[frm_indexes1]
        g_obj_boxes=gt_boxes[frm_indexes1]
        g_obj_masks=gt_masks[frm_indexes1]

        frm_indexes2=np.where(dt_frm_ids==frm_id)[0]
        d_obj_ids=dt_obj_ids[frm_indexes2]
        d_obj_boxes=dt_boxes[frm_indexes2]
        d_obj_masks=dt_masks[frm_indexes2]
        
        for k_id in xrange(len(d_obj_ids)):
            d_obj_id=d_obj_ids[k_id]
            f_index=np.where(g_obj_ids==d_obj_id)[0]
            if len(f_index)==0:
                miss_gt_cnt=miss_gt_cnt+1
            else:
                f_index=int(f_index)
                box1=g_obj_boxes[f_index]          
                mask1=g_obj_masks[f_index]
                box2=d_obj_boxes[k_id]
                mask2=d_obj_masks[k_id]

                mask_over=mask_overlap(box1,box2,mask1,mask2)

                mask_iou_arr[obj_ids_map_list[d_obj_id]]+=mask_over   ## accumulate maskover value
                mask_iou_cnt[obj_ids_map_list[d_obj_id]]+=1           ## accumulate acount
              
    ave_mask_iou_arr=mask_iou_arr/mask_iou_cnt
    ave_inst_iou=sum(mask_iou_arr)/sum(mask_iou_cnt)
    # print "========================================================"
    # ##print 'mask_iou_arr:', mask_iou_arr
    ##print 'mask_iou_cnt:', mask_iou_cnt
    # print 'ave_mask_iou_arr:', ave_mask_iou_arr
    # ##print 'det_num:', sum(mask_iou_cnt)
    # ##print 'sum_val:', sum(mask_iou_arr)
    # print 'ave_inst_iou:', ave_inst_iou
    return ave_mask_iou_arr, ave_inst_iou

