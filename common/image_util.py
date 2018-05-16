import _init_paths
from common_lib import create_dir,load_txt_to_strarr,save_strarr_to_txt
from mot_config import mcfg
import cv2
import numpy as np
import os

## save a batch of images
def save_images(ims,im_dir,im_names,im_ext):
    im_num=len(ims)
    for im_id in xrange(im_num):
        im=ims[im_id]
        im_name=im_names[im_id]
        im_path=os.path.join(im_dir,im_name+im_ext)
        cv2.imwrite(im_path,im)

def jpg_to_ppm(im_names,jpg_dir,ppm_dir):
    for im_name in im_names:
        im_path1=os.path.join(jpg_dir,im_name+'.jpg')
        im_path2=os.path.join(ppm_dir,im_name+'.ppm')
        im=cv2.imread(im_path1)
        cv2.imwrite(im_path2,im)
    print 'convert jpg to ppm...'


if __name__ == '__main__':
    data_dir=mcfg.DATA.DATA_DIR
    imgset=mcfg.DATA.IMGSET_TRAIN
    jpg_dir=mcfg.DATA.JPGDIR
    ppm_dir='MOTdevkit2017/MOT2017/PPMImages'
    im_ext=mcfg.DATA.IMGEXT

    imgset=os.path.join(data_dir,imgset)
    jpg_dir=os.path.join(data_dir,jpg_dir)
    ppm_dir=os.path.join(data_dir,ppm_dir)
    create_dir(ppm_dir)
        
    im_names=load_txt_to_strarr(imgset)
    
    im_names=im_names[:600]   ##MOT17-02
    ##print im_names
    
    print 'im_names:', im_names
    '''
    set_name='MOT17-02'
    bmf_dir='/home/uni/Lab/projects/C++/trackingCPU'
    bmf_path=os.path.join(bmf_dir,set_name+'.bmf')
    tmp_arr=[]

    for im_name in im_names:
        im_name=im_name+'.ppm'
        tmp_arr.append(im_name)

    dest_strs=np.array(tmp_arr)    
    save_strarr_to_txt(bmf_path,dest_strs)

    ppm_flag=False
    if ppm_flag:
        jpg_to_ppm(im_names,jpg_dir,ppm_dir)
    '''    
    