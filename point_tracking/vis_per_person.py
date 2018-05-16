import _init_paths
from mot_config import mcfg
from common_lib import load_color_images,load_txt_to_arr,create_dir,color_map
import os
import numpy as np
import scipy.io as sio
import cv2

## has some tricks
def load_seq_info(testset): 
    with open(testset) as f:
                  im_names = [x.strip() for x in f.readlines()]          
    f.close
    set_name='MOT16-02'
    im_names=im_names[:90]    ##MOT16-02 //trick for testing
    shift_num=510
    end_frame=600
    print 'load sequence info done...'
    return set_name,shift_num,end_frame,im_names

##
def load_proposals(res_dir, set_name):
    set_name=set_name.replace("MOT16","MOT17")   #trick due to the name inconsistency
    res_path=os.path.join(res_dir,set_name+'.txt')
    prop_arr=load_txt_to_arr(res_path)

    print 'load',len(prop_arr),'proposals done...'
    return prop_arr

def get_max_box_coord():
    print 'get max box ccordinate...'


def get_rect_point_mat():
    print 'get the rect point mat...'



## This program has some problem: need to be fixed about the frame number.
if __name__ == '__main__':
    print 'visualize point tracking per person start...' 
    ##(1) color images
    data_dir=mcfg.DATA.DATA_DIR
    imgset=mcfg.DATA.IMGSET
    jpgdir=mcfg.DATA.JPGDIR  
    im_ext=mcfg.DATA.IMGEXT
    imgset_path=os.path.join(data_dir,imgset)
    jpgdir_path=os.path.join(data_dir,jpgdir)


    ##(2) point tracking
    track_res_dir=mcfg.PTRACK.RES_DIR
    frame_interval=mcfg.PTRACK.FRAME_INTER
    sample_rate=mcfg.PTRACK.SAMPLE_RATE
    sub_dir_name=str(frame_interval)+'frm_'+str(sample_rate)+'pt'
    point_mat_path= os.path.join(track_res_dir,sub_dir_name,'pt_mat.mat')
    label_mat_path= os.path.join(track_res_dir,sub_dir_name,'lab_mat.mat')
    
    ##(3) proposals
    prop_res_dir=mcfg.PROPOSAL.RES_DIR
   
    ## result directory
    vis_dir=os.path.join(track_res_dir,sub_dir_name,'per_person')
    create_dir(vis_dir)
    

    ##at present, mainly focus on large objects
    large_obj_dir=os.path.join(vis_dir,'large_person')
    small_obj_dir=os.path.join(vis_dir,'small_person')
    create_dir(large_obj_dir)
    create_dir(small_obj_dir)
    size_thre=mcfg.PTRACK.OBJECT_SIZE_THRESHOLD


    set_name,shift_num,end_frame,im_names=load_seq_info(imgset_path)
    
    rgb_ims=load_color_images(jpgdir_path,im_names,im_ext)
 
    prop_arr=load_proposals(prop_res_dir,set_name)
    

    im_width=rgb_ims[0].shape[1]
    im_height=rgb_ims[0].shape[0]

    point_mat=sio.loadmat(point_mat_path)['X']      #point_mat.shape=(2F,D) ---> F is the framenumber and D is the (feature) point number
    
    point_number=point_mat.shape[1]


    ## draw bbox per person 
    labels=np.unique(prop_arr[:,1])

    labels=[3,19,23]  ##trick
    for label in labels[:3]:     ## trick- to control labels
        ##label_dir=os.path.join(vis_dir,str(int(label)))
        ##create_dir(label_dir)

        indexs=np.where(prop_arr[:,1]==label)[0]
        frm_nums=prop_arr[indexs,0]+shift_num
        coords=prop_arr[indexs,2:6]
        
        ##coords=coords[:10]    #trick here  
        x_min=int(max(0,min(coords[:frame_interval,0])-1)) 
        y_min=int(max(0,min(coords[:frame_interval,1])-1))
        x_max=int(min(im_width-1,max(coords[:frame_interval,0]+coords[:frame_interval,2]-1)+1))
        y_max=int(min(im_height-1,max(coords[:frame_interval,1]+coords[:frame_interval,3]-1)+1))

        w_max=x_max-x_min+1
        h_max=y_max-y_min+1

        if w_max>=size_thre:
            label_dir=os.path.join(large_obj_dir,str(int(label)))
            create_dir(label_dir)  

        else:
            label_dir=os.path.join(small_obj_dir,str(int(label)))
            create_dir(label_dir)
        
         ## based on the rect range() to get the rect_point_mat
        inner_rec_cols=[]
        
        for pt_id in xrange(point_number):
            pt_flag=True
            for frm_id in xrange(frame_interval):   
                x=int(np.round(point_mat[frm_id*2,pt_id]))
                y=int(np.round(point_mat[frm_id*2+1,pt_id]))
                if x<x_min or x>x_max or y<y_min or y>y_max:
                    pt_flag=False
            if pt_flag:
                inner_rec_cols.append(pt_id)

        inner_rec_cols = np.array(inner_rec_cols, dtype=np.int32) 
       
        rect_point_mat=point_mat[:,inner_rec_cols]

        rect_point_mat_path=os.path.join(label_dir,str(int(label))+'_rec_mat.mat')
        sio.savemat(rect_point_mat_path,{'X':rect_point_mat})


        ## for load the clustering results.
        rect_point_lab_path=os.path.join(label_dir,str(int(label))+'_rec_lab.mat')
        rect_label_mat=sio.loadmat(rect_point_lab_path)['grp']    #label_mat.shape=(D,1)  ---> D is the(feature) point number

        print 'rect_point_mat.shape:',rect_point_mat.shape
        print 'rect_label_mat.shape:',rect_label_mat.shape
        
     
        cmap = color_map()  ## color map
        color=cmap[label]
        colo= np.array((int(color[0]),int(color[1]),int(color[2])))
        print x_min,y_min,x_max,y_max
        print colo
        
        ##for frm_id in xrange(len(frm_nums)):
        deal_len=min(frame_interval,len(frm_nums))   ##trick- to control frame numbers
        for frm_id in xrange(deal_len):
        ##for frm_id in xrange(1):    
            im_name=set_name+'_'+str(int(frm_nums[frm_id])).zfill(6)+im_ext 
            vis_im_path=os.path.join(label_dir,im_name)
            im=rgb_ims[int(frm_nums[frm_id]-shift_num)]
            
            coord=coords[frm_id]
            x1=max(0,int(np.round(coord[0])))
            y1=max(0,int(np.round(coord[1])))
            x2=min(im_width-1,int(np.round(coord[0]+coord[2]))-1)
            y2=min(im_height-1,int(np.round(coord[1]+coord[3]))-1)

            im1=im.copy()
            cv2.rectangle(im1,(x_min,y_min),(x_max,y_max),(0,255,0),1)   #max_box
            cv2.rectangle(im1,(x1,y1),(x2,y2),(0,0,255),1)   # predicted proposal
            
            
            rect_pt_num=rect_point_mat.shape[1]  #point number

            ## draw clustering results
            for pt_id in xrange(rect_pt_num): ##point_number
                x=int(np.round(rect_point_mat[frm_id*2,pt_id]))
                y=int(np.round(rect_point_mat[frm_id*2+1,pt_id]))
                
                x1=max(0,x-2)
                y1=max(0,y-2)
                x2=min(x+2,im_width-1)
                y2=min(y+2,im_height-1)

                rect_label=int(rect_label_mat[pt_id])
                color=cmap[rect_label]
                colo= np.array((int(color[0]),int(color[1]),int(color[2])))
                
                cv2.rectangle(im1,(int(x1),int(y1)),(int(x2),int(y2)),(colo),2)
        
            cv2.imwrite(vis_im_path,im1)
            
            ##cv2.imshow('im',im)
            ##cv2.waitKey(-1)
                

