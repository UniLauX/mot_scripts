import _init_paths
from mot_config import mcfg
import os
import numpy as np
from common_lib import create_dir

seg_im_ext='.sm'
vis_mask_flag=True
thresh_mask=0.5
alpha=0.5
mid_line_width=2
im_ext='.jpg'
import cv2

## drawing with different color
def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])
    cmap = cmap/255 if normalized else cmap
    return cmap

## Visualize gt boxes
def vis_gt_bboxes(proposal_algr_dir,im,im_name,bboxes,labels=[]):
    print '==============================vis_gt_bboxes=============================================='
    print 'proposal_algr_dir:', proposal_algr_dir
    vis_im=im.copy()
    im_ext=mcfg.DATA.IMGEXT
    im_width=im.shape[1]
    im_height=im.shape[0] 

    print 'im_width:',im_width
    print 'im_height:',im_height   
    
    cmap=color_map()

    for n_id in xrange(0,len(bboxes)):
        bbox=bboxes[n_id]
        x1 = bbox[0]
        y1 = bbox[1]
        x2= bbox[2]
        y2= bbox[3]
        x1 = np.min((im_width - 1, np.max((0, x1))))
        y1 = np.min((im_height - 1, np.max((0, y1))))
        x2 = np.min((im_width - 1, np.max((0, x2))))
        y2 = np.min((im_height - 1, np.max((0, y2))))
        
        if len(labels)==0:
            colo=np.array((0,0,255))
        else:
            label=labels[n_id]
            color=cmap[label]
            colo= np.array((int(color[0]),int(color[1]),int(color[2])))
        #draw rectangle
        cv2.rectangle(vis_im,(int(x1),int(y1)),(int(x2),int(y2)),(colo),mid_line_width) 
     #visualize gt boxes
    ##cv2.imshow("gtboxes", vis_im)
    cv2.imwrite(os.path.join(proposal_algr_dir,im_name+im_ext),vis_im)  #complete annotation
    ##cv2.waitKey(-1) 

## vis multiple mask proposals in one image 
def vis_proposals_multi_instances(proposal_algr_dir,det_proposals_folder,im,im_name,bboxes,masks,labels):
    print '==============================vis_proposals==========================================='
    
    vis_im=im.copy()     
    im_width=im.shape[1]
    im_height=im.shape[0] 
    cmap = color_map()  ## color map
    base_im_name=im_name
   
    vis_det_dir=os.path.join(proposal_algr_dir, det_proposals_folder)
    vis_seg_dir=vis_det_dir  ## det & seg in the same folder
    create_dir(vis_det_dir)
    print '==========================vis proposals=================================' 
    print 'im_name:', im_name
    print 'im_width:',im_width
    print 'im_height:',im_height  
    print 'len(masks):',len(masks)
    print 'bboxes.shape:', bboxes.shape
    print 'seg_im_ext:', seg_im_ext
    print 'vis_det_dir:', vis_det_dir
       
    vis_im_seg=vis_im   
    for b_id in xrange(0,len(bboxes)):   ## here obj_id is actually the index(of the array)
        bbox=bboxes[b_id]
        x1 = int(round(bbox[0]))
        y1 = int(round(bbox[1]))
        x2= int(round(bbox[2]))
        y2= int(round(bbox[3]))
        x1 = np.min((im_width - 1, np.max((0, x1))))
        y1 = np.min((im_height - 1, np.max((0, y1))))
        x2 = np.min((im_width - 1, np.max((0, x2))))
        y2 = np.min((im_height - 1, np.max((0, y2))))
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        ##object id and assign color
        label=labels[b_id]
        color=cmap[label]
        colo= np.array((int(color[0]),int(color[1]),int(color[2])))
        
        # ##'----------------------------------Mask part------------------------------------------------'
        # # mask = masks[b_id, :, :]    #mask.shape=(28,28)
        # # mask = cv2.resize(mask, (int(w), int(h)), interpolation=cv2.INTER_LINEAR) #bilinear interpolation
        # mask[mask >= thresh_mask] = 1
        # mask[mask < thresh_mask] = 0
        mask=masks[b_id]
        
        ###===============================per-person visualization=======================================
        if vis_mask_flag:
            label_id_fg=(mask >= thresh_mask)
            label_mask_t = np.zeros((h, w, 3), dtype=np.uint8)
            
            label_im_box=im[y1:y2 + 1, x1:x2 + 1, :] 
            label_mask_t[:, :, :]= vis_im_seg[y1:y2 + 1, x1:x2 + 1, :]    # copy the proposal region from the color image
            
            label_mask_t[label_id_fg, 0] = colo[0]
            label_mask_t[label_id_fg, 1] = colo[1]
            label_mask_t[label_id_fg, 2] = colo[2]
            
            cv2.addWeighted(label_mask_t, alpha, label_im_box, 1 - alpha, 0, label_mask_t)
                         
            vis_im_seg[y1:y2 + 1, x1:x2 + 1, :] = label_mask_t
            ##cv2.imwrite(vis_label_mask_path,label_mask_t)
            ##cv2.imwrite(vis_label_mask_path,vis_im_seg)
            ##cv2.imshow('label_mask',label_mask_t) 
            ##cv2.waitKey(-1)
    
        cv2.rectangle(vis_im,(x1,y1),(x2,y2),(colo),mid_line_width)
        font = cv2.FONT_HERSHEY_SIMPLEX
    vis_im_path=os.path.join(vis_det_dir,im_name+im_ext)
    print 'vis_im_path:', vis_im_path
    ##cv2.imshow('det_proposal', vis_im)
    cv2.imwrite(vis_im_path,vis_im)  #complete annotation
    ##cv2.waitKey(0)   

def vis_link_imgs(im_names,src_dir1,src_dir2,dest_dir):
    create_dir(dest_dir)
    for im_name in im_names:
        im_path1=os.path.join(src_dir1,im_name+im_ext)
        im_path2=os.path.join(src_dir2,im_name+im_ext)
        link_im_path=os.path.join(dest_dir,im_name+im_ext)

        im1=cv2.imread(im_path1)
        im2=cv2.imread(im_path2)
        im_height=im1.shape[0]
        im_width=im1.shape[1]
       
        link_im= np.zeros((im_height*2, im_width, 3), dtype=np.uint8)
       
        link_im[0:im_height,:,:]=im1
        link_im[im_height:im_height*2,:,:]=im2
        cv2.imwrite(link_im_path,link_im)
        ##print '=================vis_link_img===================='


##======================================= Vis per person==========================================================
## Visualize gt boxes
def vis_box_per_person(vis_dir,ims,im_names,bboxes,label,masks,bboxes1=[]):
    print '==============================vis_gt_bboxes=============================================='
    print 'vis_dir:', vis_dir
    im_num=len(ims)
    im_ext=mcfg.DATA.IMGEXT

    colo_white=np.array((255,255,255)) 
    cmap=color_map()
    color=cmap[label]
    colo= np.array((int(color[0]),int(color[1]),int(color[2])))
    
    for n_id in xrange(im_num):
        im=ims[n_id]
        im_name=im_names[n_id]
        vis_im=im.copy()
        im_width=im.shape[1]
        im_height=im.shape[0] 
        ##print 'im_width:',im_width
        ##print 'im_height:',im_height   
        bbox=bboxes[n_id]
        x1 = int(np.round(bbox[0]))
        y1 = int(np.round(bbox[1]))
        x2= int(np.round(bbox[2]))
        y2= int(np.round(bbox[3]))
        x1 = np.min((im_width - 1, np.max((0, x1))))
        y1 = np.min((im_height - 1, np.max((0, y1))))
        x2 = np.min((im_width - 1, np.max((0, x2))))
        y2 = np.min((im_height - 1, np.max((0, y2))))
        h=y2-y1+1
        w=x2-x1+1
        tmp_mask=masks[n_id]
        mask=tmp_mask[y1:y2+1,x1:x2+1]
          ###===============================per-person visualization=======================================
        if vis_mask_flag:
            label_id_fg=(mask >= thresh_mask)
            label_mask_t = np.zeros((h, w, 3), dtype=np.uint8)
            
            label_im_box=im[y1:y2 + 1, x1:x2 + 1, :] 
            label_mask_t[:, :, :]= vis_im[y1:y2 + 1, x1:x2 + 1, :]    # copy the proposal region from the color image
            
            label_mask_t[label_id_fg, 0] = colo[0]
            label_mask_t[label_id_fg, 1] = colo[1]
            label_mask_t[label_id_fg, 2] = colo[2]
            
            cv2.addWeighted(label_mask_t, alpha, label_im_box, 1 - alpha, 0, label_mask_t)
                         
            vis_im[y1:y2 + 1, x1:x2 + 1, :] = label_mask_t
        #draw rectangle
        cv2.rectangle(vis_im,(x1,y1),(x2,y2),(colo),mid_line_width) 
        if len(bboxes1)>0:
            box1=bboxes1[n_id]
            cv2.rectangle(vis_im,(int(box1[0]),int(box1[1])),(int(box1[2]),int(box1[3])),(colo_white),mid_line_width) 
     #visualize gt boxes
    ##cv2.imshow("gtboxes", vis_im)
        cv2.imwrite(os.path.join(vis_dir,im_name+im_ext),vis_im)  #complete annotation
    ##cv2.waitKey(-1) 
    

