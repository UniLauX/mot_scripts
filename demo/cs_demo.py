import _init_paths
from cs_config import cscfg
import os
import cv2
import numpy as np
from common_lib import create_dir
from video_util import im_seq_to_video 


def prepare_imgset():
    ## parameters
    data_root_dir=cscfg.DATA.ROOT_DIR
    set_branch=cscfg.DATA.BRANCH      ## validation set 
    sub_set_dir=os.path.join(data_root_dir,set_branch)
    imgset_dir=cscfg.DATA.IMGSET_DIR
    im_ext=cscfg.DATA.IM_EXT
    
    if set_branch=='val':
        set_names=['frankfurt','lindau','munster']
    
    imgset_dir=os.path.join(imgset_dir,set_branch)
    create_dir(imgset_dir)

    for set_name in set_names:
        print set_name
        imgset_path=os.path.join(imgset_dir,set_name+'.txt')
        arr=os.listdir(os.path.join(sub_set_dir,set_name))
        arr=np.sort(arr)
        print arr
        print imgset_path

        with open(imgset_path, 'w') as f1:
            for row_id in xrange(0, len(arr)):
                im_name=arr[row_id][:-4]
                f1.write('{:s}\n'. format(im_name)) 
                 
        f1.close  
          
def get_video_demo():

    det_res_dir=cscfg.DETCTION.RES_DIR
    det_algr=cscfg.DETCTION.ALG
    ##det_im_dir=det_res_dir
    det_im_dir=os.path.join(det_res_dir,det_algr) 
    im_ext=cscfg.DATA.IM_EXT

    data_branch=cscfg.DATA.BRANCH  
    imgset_dir=cscfg.DATA.IMGSET_DIR
    imgset_dir=os.path.join(imgset_dir,data_branch)
    
    video_demo_dir=os.path.join(det_res_dir,'video_demo')
 
    if data_branch=='val':
        set_names=['frankfurt','lindau','munster','frankfurt_lindau_munster']
    ##if data_branch=='train'
    ##if data_branch=='test'
   
    video_fps=cscfg.DEMO.VIDEO_FPS

    for set_name in set_names[3:4]: ## at present,just in 'frankfurt' dataset
        imgset_path=os.path.join(imgset_dir,set_name+'.txt') 
        det_im_dir=os.path.join(det_im_dir,set_name)
        print imgset_path
        print det_im_dir
        print video_demo_dir
 
        with open(imgset_path) as f:
                  im_names = [x.strip() for x in f.readlines()]          
        f.close

        im_names=im_names

        im_seq_to_video(det_algr, det_im_dir,im_names,im_ext,video_demo_dir,video_fps)
    print 'get video based on detection results...'


if __name__ == '__main__':
    ##prepare_imgset()       #func1
    get_video_demo()       #func2   
   
    

   
    
    