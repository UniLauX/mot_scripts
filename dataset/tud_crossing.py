import _init_paths
import numpy as np
import os
import cv2
from common_lib import load_txt_to_fltarr, get_mini_box_from_mask
from common_lib import create_dir
from box_util import coords_to_boxes
from mot_util import write_detarr_as_mot  ## org
from vis_util import vis_gt_bboxes

debug_flag=True
##data_dir='/mnt/phoenix_fastdir/dataset/TUD-Crossing'
data_dir='/media/uni/uni2/dataset/TUD-Crossing'
mot_data_dir='/mnt/phoenix_fastdir/dataset/backup/MOT/motchallenge-devkit/data/MOT15/test/TUD-Crossing'
new_gt_folder='gt_new'




##================================================================================================
## tud_crossing format:
## dets: frame,x1,y1,w,h,visibility, ASC(l),ASC(r),ASC(h),ASC(f),ASC(b), uniqueID
## segs: ...
## get org dets annots path
def get_dets_annots_path(data_dir):
    dets_annots_path=os.path.join(data_dir,'tud-crossing-sequence_segmentation_ICG_anno.txt')
    return dets_annots_path

## get org segs annots dir
def get_segs_annots_dir(data_dir):
    segs_annots_dir=os.path.join(data_dir,'tud-crossing-sequence_segmentation_ICG_segs')
    return segs_annots_dir

## load org gt_dets from .txt
def load_dets_annots(dets_annots_path):
    dets_arr=load_txt_to_fltarr(dets_annots_path)
    return dets_arr

## load org gt_segs from .png
def load_segs_annots(segs_annots_dir):
    segs_annots_dir=get_segs_annots_dir(data_dir)
    seg_im_names=[]
    seg_gt_ims=[]
    for filename in os.listdir(segs_annots_dir):
        if filename.find('_seg.png')!=-1:
            seg_im_names.append(filename)
    seg_im_names=np.sort(seg_im_names)
    
    for seg_im_name in seg_im_names:
        seg_im_path=os.path.join(segs_annots_dir,seg_im_name)
        seg_im=cv2.imread(seg_im_path)  
        ##obj_ids=np.unique(seg_im)
        seg_gt_ims.append(seg_im)      

    return seg_gt_ims  # for every image, shape=(h,w,c)

## parse org gt dets from .txt (tud-crossing)
def get_org_gt_dets():
    dets_annots_path=get_dets_annots_path(data_dir)
    dets_arr=load_dets_annots(dets_annots_path)
    frm_ids=dets_arr[:,0].astype(int)
    obj_ids=dets_arr[:,-1].astype(int)
    coords=dets_arr[:,1:5]
    boxes=coords_to_boxes(coords)
    return frm_ids,obj_ids,boxes

## load gt segs from .jpg (tud-crossing),##sort accoring to frm_no
def get_org_gt_masks():    
    segs_annots_path=get_segs_annots_dir(data_dir)
    segs_arr=load_segs_annots(segs_annots_path)
    return segs_arr

# org_name: prepare_gt_masks
## prepare gt_binary_mask from png_gt_mask and/or not with gt_boxes
def prepare_binary_gt_masks(gt_frm_ids,gt_obj_ids,gt_masks,gt_boxes=[],tight_box_flag=False):
    im_height,im_width=gt_masks[0].shape[:2]
    
    #masks=[]
    masks=np.empty((0,im_height,im_width))
    boxes=np.empty((0,4))
    
    uniq_frm_ids=np.sort(np.unique(gt_frm_ids))
    # ## calculate based on per-frame
    for n_id in xrange(len(uniq_frm_ids)):
        u_id=uniq_frm_ids[n_id]   ##frm_id
        g_row_indexes=np.where(gt_frm_ids==u_id)[0]

        g_obj_ids=gt_obj_ids[g_row_indexes]      
        g_mx_mask=gt_masks[n_id]      ## rgb_gt_mask

        if tight_box_flag:
            g_obj_masks,g_obj_boxes=convert_binary_masks(g_mx_mask,g_obj_ids)
            ##print g_obj_masks.shape
        else:
            g_obj_boxes=gt_boxes[g_row_indexes].astype(int) ## org gt_box was not tight
            g_obj_masks=convert_binary_masks1(g_mx_mask,g_obj_ids,g_obj_boxes)
    
        ##masks=np.hstack((masks,g_obj_masks))
        masks=np.concatenate((masks,g_obj_masks), axis=0)
        boxes=np.concatenate((boxes,g_obj_boxes), axis=0)     
    return boxes,masks  

## convert gt_mask from .jpg(in one image) to arrays(several images) 
## from multi-labels to binary labels : tight_box
def convert_binary_masks(mask_img,obj_ids):
    obj_num=len(obj_ids)
    if obj_num==0:
        return []
    m_img=mask_img[:,:,0] ## im_shape=(im_height, im_width)
    obj_masks=[]
    obj_boxes=[]

    for n_id in xrange(len(obj_ids)):
    ##for n_id in xrange(1):
        tmp_mask=np.zeros((mask_img.shape[0],mask_img.shape[1]))
        obj_id=obj_ids[n_id]        
        m_ids=(m_img==obj_id)
        tmp_mask[m_ids]=1
        # print np.max(tmp_mask)
        # cv2.imshow('tmp_mask',tmp_mask*128)
        # cv2.waitKey(-1)
        mini_box=get_mini_box_from_mask(tmp_mask)
        mini_box=np.asarray(mini_box).astype(int)  # type-trick 
        ##mask=tmp_mask[mini_box[1]:mini_box[3]+1,mini_box[0]:mini_box[2]+1]
        mask=tmp_mask   ## shape=(im_height, im_width)
        # print mini_box.shape
        # print mask.shape
        obj_masks.append(mask)
        obj_boxes.append(mini_box)

    ## continue from here, has some problem...
    obj_masks=np.squeeze(np.asarray(obj_masks))
    obj_boxes=np.squeeze(np.asarray(obj_boxes))
    # print obj_masks.shape
    # print obj_boxes.shape
    return obj_masks,obj_boxes

## convert gt_mask from .jpg(in one image) to arrays(several images) 
## from multi-labels to binary labels: using provieded boxes
def convert_binary_masks1(mask_img,obj_ids,boxes):
    obj_num=len(obj_ids)
    if obj_num==0:
        return []  
    obj_masks=[]
    m_img=mask_img[:,:,0]  ## shape=(im_height,im_width)
    for n_id in xrange(len(obj_ids)):
        tmp_mask=np.zeros((mask_img.shape[0],mask_img.shape[1]))
        obj_id=obj_ids[n_id]
        box=boxes[n_id]       
        m_ids=(m_img==obj_id)
        tmp_mask[m_ids]=1
        ##cv2.imshow('tmp_mask',tmp_mask*128)
        ##cv2.waitKey(-1)
        ##mask=tmp_mask[box[1]:box[3]+1,box[0]:box[2]+1]
        mask=tmp_mask ## shape=(im_height, im_width)
        obj_masks.append(mask)
        # print box.shape
        # print mask.shape
    obj_masks=np.asarray(obj_masks)
    return obj_masks

##------------------------------------save gt_new--------------------------------------------------
##================================================================================================
def save_masks_new(gt_masks_tensor):
    gt_masks_path=os.path.join(data_dir,new_gt_folder,'masks')
    print gt_masks_path
    print gt_masks_tensor.shape
    np.save(gt_masks_path,gt_masks_tensor,allow_pickle=False)
    ##np.save(gt_masks_path,gt_masks_tensor,allow_pickle=True)

## load provided gt_masks
def load_masks_new():
    gt_masks_path=os.path.join(data_dir,new_gt_folder,'masks.npy')
    print gt_masks_path
    gt_masks_tensor=np.load(gt_masks_path)
    return gt_masks_tensor

def save_dets_new(gt_dets_arr):
    gt_dets_path=os.path.join(data_dir,new_gt_folder,'dets.txt')
    write_detarr_as_mot(gt_dets_path,gt_dets_arr)
    print 'save dets new...'

## load dets that with tight_gt_boxes
def load_dets_new():
    gt_dets_path=os.path.join(data_dir,new_gt_folder,'dets.txt')
    gt_dets=load_txt_to_fltarr(gt_dets_path)
    return gt_dets

## save dets that with lose_gt_boxes
def save_dets_org(gt_dets_arr):
    gt_dets_path=os.path.join(data_dir,new_gt_folder,'dets_lose.txt')
    write_detarr_as_mot(gt_dets_path,gt_dets_arr)
    print 'save dets org...'

## load dets that with lose_gt_boxes
def load_dets_org():
    gt_dets_path=os.path.join(data_dir,new_gt_folder,'dets_lose.txt')
    gt_dets=load_txt_to_fltarr(gt_dets_path)
    return gt_dets

##-----------------------------------external-----------------------------------------
##================================================================================================
## mot15 format:
## dets: frame,uniqueID,x1,y1,w,h,scores(=1 for gt),x,y,z 
## segs: empty
def dets_tud_to_mot(tud_dets_path,mot_dets_path):
    tud_dets_arr=load_dets_annots(tud_dets_path)
    frm_ids=tud_dets_arr[:,0].astype(int)
    obj_ids=tud_dets_arr[:,-1].astype(int)
    coords=tud_dets_arr[:,1:5]
    ##vis_arr=tud_dets_arr[:,5]/100.0   ##make vis range in [0, 1.0]
    boxes=coords_to_boxes(coords)   ## unnormal
    row_num=len(frm_ids)
    mot_dets_arr=np.ones((row_num,10))*-1
 
    mot_dets_arr[:,0]=frm_ids
    mot_dets_arr[:,1]=obj_ids
    mot_dets_arr[:,2:6]=coords
    mot_dets_arr[:,6]=1

    ##write_detarr_as_mot(mot_dets_path,mot_dets_arr)  ## org
    if debug_flag:
        # print frm_ids.shape
        # print obj_ids.shape
        # print obj_ids
        print mot_dets_arr
    #     print  np.unique(frm_ids)
    #     print  np.unique(obj_ids)
    #     print  boxes.shape
    #     ##print  coords
# mot_data_dir= /motchallenge.../TUD-Crossing             
def get_mot_gt_path(mot_data_dir):
    dets_gt_dir=os.path.join(mot_data_dir,'gt')
    create_dir(dets_gt_dir)
    dets_gt_path=os.path.join(dets_gt_dir,'gt.txt')
    return dets_gt_path

def get_mot_img_dir(mot_data_dir):
    im1_dir=os.path.join(mot_data_dir,'img1')
    return im1_dir


if __name__ == '__main__':
    print '=========================tud_crossing to MOT==================================='
    tud_dets_gt_path=get_dets_annots_path(data_dir) 
    mot_dets_gt_path=get_mot_gt_path(mot_data_dir)  
    dets_tud_to_mot(tud_dets_gt_path,mot_dets_gt_path)
  





