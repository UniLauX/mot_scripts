import _init_paths
import os
import cv2
import numpy as np
from common_lib import create_dir
from video_util import im_seq_to_video 


def prepare_imgset():
    ## parameters
    cs_val_dir='/mnt/phoenix_fastdir/dataset/cityscapes/leftImg8bit_sequence/val'
    imgset_dir='/mnt/phoenix_fastdir/dataset/cityscapes/leftImg8bit_sequence/imgset'
    set_names=['frankfurt','lindau','munster']
    
    imgset_val_dir=os.path.join(imgset_dir,'val')
    create_dir(imgset_val_dir)

    name_suffix='leftImg8bit'
    im_ext='.png'
    
    for set_name in set_names:
        imgset_val_path=os.path.join(imgset_val_dir,set_name+'.txt')
        arr=os.listdir(os.path.join(cs_val_dir,set_name))
        arr=np.sort(arr)
        print arr
        print imgset_val_path
        
        with open(imgset_val_path, 'w') as f1:
            for row_id in xrange(0, len(arr)):
                im_name=arr[row_id][:-4]
                f1.write('{:s}\n'. format(im_name)) 
                 
        f1.close  
          

if __name__ == '__main__':
    ##prepare_imgset()
    im_seq_to_video(rgb_img_dir,im_names,dest_video_path)
   
