import _init_paths
from mot_config import mcfg
from common_lib import create_dir
from video_util import im_seq_to_video

import os    

def get_video_demo():
    print 'get video demo...'

if __name__ == '__main__':
    get_video_demo()

    ## MOT
    dataset='MOT17'

    det_res_dir=mcfg.DETCTION.RES_DIR 
    det_res_dir=os.path.join(det_res_dir,dataset)
    det_algr=mcfg.DETCTION.ALG
    det_im_dir=os.path.join(det_res_dir,det_algr) 
    video_fps=mcfg.DEMO.VIDEO_FPS
     
    ##in local disk(transfer to phoenix later)
    data_dir=mcfg.DATA.DATA_DIR
    imgset_path=mcfg.DATA.IMGSET='MOTdevkit2016/MOT2016/ImageSets/Main/val.txt'
    imgset_path=os.path.join(data_dir,imgset_path)
    im_ext=mcfg.DATA.IMGEXT='.jpg'
   
    video_demo_dir=os.path.join(det_res_dir,'video_demo')
    if not os.path.exists(video_demo_dir):
        create_dir(video_demo_dir)

    with open(imgset_path) as f:
                im_names = [x.strip() for x in f.readlines()]          
    f.close
    im_names=im_names
    
    ##video_demo_dir=os.path.join('/mnt/phoenix_fastdir/experiments/detection/MOT','video_demo')
    im_seq_to_video(det_algr, det_im_dir,im_names,im_ext,video_demo_dir,video_fps)
        

