from common_lib import load_txt_to_fltarr
from box_util import coords_to_boxes,boxes_to_coords
from mot_util import write_detarr_as_mot
import numpy as np

debug_flag=True

## find the (most nearby) lower id and upper id. eg m_index =5,  indexes= 1,3, 9,10. then the l_pos=3, u_pos=9
def find_two_nearby_positions(m_index,indexes,cont_indexes):
    ## get positions in disc_indexes
    tmp_indexes=indexes-m_index
    neg_indexes= np.where(tmp_indexes<0)[0]    
    l_pos=len(neg_indexes)-1
    u_pos=l_pos+1

    ## map from disc_indexes to cont_indexes
    l_ele=indexes[l_pos]
    u_ele=indexes[u_pos]
    l_pos=np.where(cont_indexes==l_ele)[0]
    u_pos=np.where(cont_indexes==u_ele)[0]
    return l_pos, u_pos

## add ele into array
def add_element_to_array(ele, array):
    tmp_array=np.concatenate(([ele],array))
    sort_array=np.sort(tmp_array)
    return sort_array

## interface
def remove_element_from_array(ele,array):
    print 'remove element from arrary...'

## get continious boxes from discrete boxes
def get_cont_boxes(disc_boxes,disc_frm_ids,miss_frm_ids):
    box_num=len(disc_frm_ids)+len(miss_frm_ids)
    cont_boxes=np.zeros((box_num,4))    

    ## all indexes is based on cont_indexes
    start_frm_id=min(disc_frm_ids)
    disc_indexes=disc_frm_ids-start_frm_id   ## make all the indexes start from 0
    miss_indexes=miss_frm_ids-start_frm_id
    cont_indexes=np.arange(box_num)

    ## case1: assign discrete_boxes
    cont_boxes[disc_indexes,:]=disc_boxes   

    ## case2: assign miss_boxes
    for m_index in miss_indexes:
        low_pos,up_pos=find_two_nearby_positions(m_index,disc_indexes,cont_indexes)
        insert_box=(cont_boxes[low_pos]+cont_boxes[up_pos])/2.0  ## interploate box 
        cont_boxes[m_index]=insert_box

        disc_indexes=add_element_to_array(m_index,disc_indexes) # move the obj_id into discrete_indexes

    # if debug_flag:
    #     print cont_boxes
    return cont_boxes
 
##--------------------------------------- indexes---------------------------------------------------------
## comp_arr is continious, sub_arr is discrete 
def get_miss_indexes(comp_arr, sub_arr):
    miss_arr=[]
    tmp_arr=set(list(comp_arr))-set(list(sub_arr))
    miss_arr=np.array(list(tmp_arr))
    return miss_arr


## interploate base on obj_id(for a specific instance of a sequence)
def interploate_label_dets(obj_id,disc_det_arr):
    cont_det_arr=[]
    tmp_det_arr=[]
    
    col=0           ## frame_no
    sort_det_arr=disc_det_arr[np.argsort(disc_det_arr[:,col])]  ## sort according to obj_id
    
    disc_frm_ids=sort_det_arr[:,col].astype(int)

    disc_boxes=sort_det_arr[:,2:6] ## x1,y1,x2, y2 
    
    # print 'disc_frm_ids:', disc_frm_ids

    if len(disc_frm_ids)==1:             ## case1: one detection 
        cont_det_arr=np.asarray(disc_det_arr)
        print cont_det_arr.shape
        return cont_det_arr
    else:
        cont_frm_ids=np.arange(min(disc_frm_ids),max(disc_frm_ids)+1)
        miss_frm_ids=get_miss_indexes(cont_frm_ids,disc_frm_ids)
        if len(miss_frm_ids)==0:        ## case2: continious detection         
            cont_det_arr=np.asarray(disc_det_arr)  
            print cont_det_arr.shape
            return cont_det_arr
        else:                           ## case3: need to interploate detectlets
            cont_det_arr=np.zeros((len(cont_frm_ids),disc_det_arr.shape[1]))   # shape-(n_rows, 7)
            start_frm_id=min(disc_frm_ids)
            disc_indexes=disc_frm_ids-start_frm_id  #make index start from 0
            miss_indexes=miss_frm_ids-start_frm_id

            ## contious det(label) proposals
            cont_boxes=get_cont_boxes(disc_boxes,disc_frm_ids,miss_frm_ids)
            ##cont_coords= boxes_to_coords(cont_boxes) ## x1,y1,x2,y2-> x1,y1,w,h
            
            cont_det_arr[:,0]=cont_frm_ids  ## assign frm_ids
            cont_det_arr[:,1]=obj_id       ## obj_id
            cont_det_arr[:,2:6]=cont_boxes ## assign boxes
            
            # cont_det_arr[disc_indexes,6]=disc_det_arr[:,6]  ## copy org scores(from mask-rcnn)
            # cont_det_arr[miss_indexes,6]=interploate_score  ## can be adjusted(=1.0 now)
           
            cont_det_arr=np.asarray(cont_det_arr)

            # if debug_flag:
            #     # print disc_indexes
            #     # print miss_indexes
            #     print cont_det_arr
    return cont_det_arr

## interploate base on a set name( for all instance of a sequence)
def interploate_dets(disc_dets_path,cont_dets_path):
    print '==================interploate det proposals=========================='
    print '--------------------------discrete_det_arr--------------------------------------------'

    disc_det_arr=load_txt_to_fltarr(disc_dets_path)
    #print 'disc_det_arr:\n', disc_det_arr    
    tmp_det_arr=disc_det_arr  # frm-no, obj_id, x1,y1,w,h
    tmp_coords=tmp_det_arr[:,2:6]  ## x1,y1,w,h
    tmp_boxes=coords_to_boxes(tmp_coords) ## x1,y1,x2, y2
    tmp_det_arr[:,2:6]=tmp_boxes

    print '--------------------------------------------------------------------'
    #frm_no,obj_id,x1,y1,x2,y2
    col=1  ## obj_id (col)
    sort_det_arr=tmp_det_arr[np.argsort(tmp_det_arr[:,col])]  ## sort according to obj_id
    obj_ids=np.unique(sort_det_arr[:,1]).astype(int)

    print '----------------------------cont_det_arr-------------------------------------------' 
    cont_det_arr=[]
    tmp_cont_det_arr=np.zeros((0,sort_det_arr.shape[1]))

    ## interploate box based on each obj_id    
    for obj_id in obj_ids:
        #print '--------------------------obj_id=', obj_id, '------------------------------------------------'
        row_indexes= np.where(sort_det_arr[:,1]==obj_id) 
        disc_label_det_arr=sort_det_arr[row_indexes]
        cont_label_det_arr= interploate_label_dets(obj_id,disc_label_det_arr)
        
        # print 'disc_label_det_arr.shape:', disc_label_det_arr.shape
        # print 'cont_label_det_arr.shape:', cont_label_det_arr.shape
        if disc_label_det_arr.shape == cont_label_det_arr.shape:
            print 'obj_id that equal:', obj_id

        cont_label_boxes=cont_label_det_arr[:,2:6]             ## convert back from boxes to coords
        cont_label_coords=boxes_to_coords(cont_label_boxes)
        cont_label_det_arr[:,2:6]=cont_label_coords 
        tmp_cont_det_arr=np.vstack((tmp_cont_det_arr,cont_label_det_arr))
    
    cont_det_arr=np.asarray(tmp_cont_det_arr)    #convert list to array
    write_detarr_as_mot(cont_dets_path,cont_det_arr)  ## save in local disk as .txt
    

    if debug_flag:
        print 'cont_det_arr.shape:', cont_det_arr.shape   
    

        




