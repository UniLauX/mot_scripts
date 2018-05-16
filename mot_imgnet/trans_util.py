import _init_paths
import numpy as np
import os

import cv2

## doesn't keep orginal data
def rename_move_files(src_dir,dest_dir,org_pattern,rep_pattern):
    file_names=os.listdir(src_dir)
    for file_name in file_names:
        new_file_name=rep_pattern+file_name
        os.rename(os.path.join(src_dir,file_name),os.path.join(dest_dir,new_file_name))
       


def rename_copy_images(src_dir,dest_dir,seq_name):
    im_names=os.listdir(src_dir)
    im_names=np.sort(im_names)
    rep_pattern=seq_name+'_'
    for im_name in im_names:
        src_im_path=os.path.join(src_dir,im_name)
        dest_im_path=os.path.join(dest_dir,rep_pattern+im_name)
        im=cv2.imread(src_im_path)
        cv2.imwrite(dest_im_path,im)
    print seq_name, 'seqence with',len(im_names), 'images copyed done...'
         

def rename_create_ppmimgs(src_dir,dest_dir,seq_name):
    im_names=os.listdir(src_dir)
    im_names=np.sort(im_names)
    rep_pattern=seq_name+'_'
    for im_name in im_names:
        src_im_path=os.path.join(src_dir,im_name)
        tmp_im_path=os.path.join(dest_dir,rep_pattern+im_name)
        dest_im_path=tmp_im_path.replace('.jpg','.ppm')
        im=cv2.imread(src_im_path)
        cv2.imwrite(dest_im_path,im)
    print seq_name, 'seqence with',len(im_names), 'images copyed done...'
  