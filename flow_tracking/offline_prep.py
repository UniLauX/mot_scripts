
##(6) propagate masks based on (forward and backward) flow -> and save to local disks   
##1> input: rgb_ims(frm_num), init_masks(frm_num), init_bboxes(frm_num), fw_flows(frm_num-1), bw_flows(frm_num-1)
##2> output: mask_mat(frm_num,frm_num), bbox_mat(frm_num,frm_num), center_shift_vec(frm_num-1), maskIoU_mat(frm_num,frm_num,frm_num-1) 