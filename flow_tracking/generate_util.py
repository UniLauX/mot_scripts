import _init_paths
import os
import numpy as np
from mask_util import save_binary_mask
mask_im_ext='.sm'
split_sign=','

def save_picked_masks(iter_label_dir,set_name,picked_masks,frm_ids):
    frm_num=len(frm_ids)
    for n_id in xrange(frm_num):
        im_name=set_name+'_'+str(int(frm_ids[n_id])).zfill(6)
        mask_im_path=os.path.join(iter_label_dir,im_name+mask_im_ext)  ## mask results from MRCNN
        print mask_im_path
        mask=picked_masks[n_id]
        save_binary_mask(mask_im_path,mask)

def write_detarr_as_mot(file_path,det_float_arr):
    print '============================write detarr as mot===================================================='
    det_str_arr=det_float_arr.astype(str)
    col_num=det_str_arr.shape[1]

    with open(file_path, 'w') as f:
        for str_row in det_str_arr:
            tmp_row=str_row[0]
            for col_id in xrange(1,col_num):
                tmp_row=tmp_row+split_sign+str_row[col_id]     
            f.write('{:s}\n'. format(tmp_row))
    f.close()         
def save_picked_proposals(iter_label_prop_path,frm_ids,label,picked_bboxes):
    print 'save picked proposals....' 
    frm_num=len(frm_ids)
    det_arr=np.zeros((frm_num,6))
    for n_id in xrange(frm_num):
        box=picked_bboxes[n_id]
        frm_id=frm_ids[n_id]
        det_arr[n_id,0]=frm_id
        det_arr[n_id,1]=label
        det_arr[n_id,2:6]=box
    write_detarr_as_mot(iter_label_prop_path,det_arr)    
    # print iter_label_prop_path
    # print frm_ids
    # print label
    # print picked_bboxes
    # print det_arr

