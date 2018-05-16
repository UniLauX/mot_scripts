import _init_paths
import numpy as np
import os
import cv2

from common_lib import load_txt_to_fltarr
from flowlib import read_flo_file,visualize_flow

def get_im_names(imgset): 
    with open(imgset) as f:
                  im_names = [x.strip() for x in f.readlines()]          
    f.close
    im_names=np.asarray(im_names)
    print 'get',len(im_names),'image names...'
    return im_names


#load color images from local disk
def load_color_images(im_dir,im_names,im_ext):
    ims=[]
    for im_name in im_names:
        im_path=os.path.join(im_dir,im_name+im_ext)
        if not os.path.exists(im_path):
            print im_path,'not exist'
            break
        im=cv2.imread(im_path)
        ims.append(im)
    ims=np.asarray(ims)            ## list to array
    # cv2.imshow('im',ims[0])
    # cv2.waitKey(-1)
    print 'load', len(ims), 'color images done...'
    return ims

def load_det_proposals(prop_file_path):
    if not os.path.exists(prop_file_path):
        print prop_file_path, 'does not exist...'
        return
    prop_arr=load_txt_to_fltarr(prop_file_path)
    print 'load',len(prop_arr),'det proposals done...'
    return prop_arr



def load_opticalflows_flo(flow_dir,im_names,flo_im_ext):
    forwardflows=[]
    backwardflows=[]
    fw_flow_arr=[]
    bw_flow_arr=[]
    fw_flow_dir=os.path.join(flow_dir,'fw')
    bw_flow_dir=os.path.join(flow_dir,'bw')
    for flow_id in xrange(len(im_names)-1):       
        flow_name=im_names[flow_id]+flo_im_ext
        fw_flow_path=os.path.join(fw_flow_dir,flow_name)
        bw_flow_path=os.path.join(bw_flow_dir,flow_name)
        fw_flow=read_flo_file(fw_flow_path)  ##fw_flow.shape=(1080, 1920, 2)
        bw_flow=read_flo_file(bw_flow_path)
        forwardflows.append(fw_flow)
        backwardflows.append(bw_flow)

        fw_flow_arr=np.asarray(forwardflows)   ## list to array
        bw_flow_arr=np.asarray(backwardflows) 
        # if flow_id==0:
        #     visualize_flow(fw_flow)
    print 'load',len(forwardflows),'forward and backward flow files done...'
    return fw_flow_arr,bw_flow_arr



