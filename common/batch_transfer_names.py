import _init_paths
from flowlib import read_flow, write_flow
from mot_config import mcfg
from common_lib import create_dir,load_txt_to_strarr,save_strarr_to_txt
import os
import numpy as np
import cv2

def flow_name_transfer(src_dir,dest_dir,file_num,im_ext):

    src_fw_pref='ForwardFlow'
    src_bw_pref='BackwardFlow'

    dest_fw_pref='fw'
    dest_bw_pref='bw'
    set_name='MOT16-02'
    base_num=511

    for file_id in xrange(file_num):
        #src & dest
        src_num_name=str(file_id).zfill(3)
        dest_num_name=str(file_id+base_num).zfill(6)
        
        #fw
        src_fw_name=src_fw_pref+src_num_name+im_ext
        src_fw_path=os.path.join(src_dir,src_fw_name)
      
        dest_fw_name=dest_fw_pref+'_'+set_name+'_'+dest_num_name+im_ext
        dest_fw_path=os.path.join(dest_dir,dest_fw_name)
        
        fw_flow=read_flow(src_fw_path)
        write_flow(fw_flow,dest_fw_path)
        
        #bw
        src_bw_name=src_bw_pref+src_num_name+im_ext
        src_bw_path=os.path.join(src_dir,src_bw_name)
      
        dest_bw_name=dest_bw_pref+'_'+set_name+'_'+dest_num_name+im_ext
        dest_bw_path=os.path.join(dest_dir,dest_bw_name)

        bw_flow=read_flow(src_bw_path)
        write_flow(bw_flow,dest_bw_path)

        print src_bw_path
        print dest_bw_path
        
    print 'flow_name_transfer...'


def batch_rename_imgs(src_im_names,dest_im_names,src_im_dir,dest_im_dir,im_ext):
    
    for im_id in xrange(len(src_im_names)):
        src_im_path=os.path.join(src_im_dir,src_im_names[im_id]+im_ext)
        dest_im_path=os.path.join(dest_im_dir,dest_im_names[im_id]+im_ext)
        im=cv2.imread(src_im_path)
        cv2.imwrite(dest_im_path,im)       
    print len(src_im_names), ' batch rename images...'

def batch_rename_strarr(src_path,dest_path):
    
    src_strs=load_txt_to_strarr(src_path)
    tmp_strs=[]

    for src_str in src_strs:
        dest_str=src_str.replace('MOT16','MOT17')
        tmp_strs.append(dest_str)
        
    dest_strs=np.array(tmp_strs)    
    
    save_strarr_to_txt(dest_path,dest_strs)

    return src_strs,dest_strs

def batch_rename_func(cur_dir):
    for filename in os.listdir(cur_dir):
        print filename
        new_filename=filename.replace('MOT16','MOT17')
        os.rename(os.path.join(cur_dir,filename),os.path.join(cur_dir,new_filename))
        print new_filename
    
def motdevkit_16_to_17():
    src_dir='/mnt/phoenix_fastdir/dataset/MOTdevkit2016/MOT2016'
    dest_dir='/mnt/phoenix_fastdir/dataset/MOTdevkit2017/MOT2017'
    im_ext='.jpg'

    ##imgsets
    src_imgsets_dir=os.path.join(src_dir,'ImageSets/Main')
    dest_imgsets_dir=os.path.join(dest_dir,'ImageSets/Main')
    
    imgset_name='val.txt'
    src_imgsets_path=os.path.join(src_imgsets_dir,imgset_name)
    dest_imgsets_path=os.path.join(dest_imgsets_dir,imgset_name)

    ##jpg images
    src_jpg_dir=os.path.join(src_dir,'JPEGImages')
    dest_jpg_dir=os.path.join(dest_dir,'JPEGImages')
    create_dir(dest_jpg_dir)

    ##batch imgsets
    src_strs,dest_strs=batch_rename_strarr(src_imgsets_path,dest_imgsets_path)
     
    ##batch jpgimgs
    ##batch_rename_imgs(src_strs,dest_strs,src_jpg_dir,dest_jpg_dir,im_ext)

    ##'/Annotations'    
    ##cur_dir='/mnt/phoenix_fastdir/dataset/MOTdevkit2017/MOT2017/Annotations'
    bw_flow_dir='/mnt/phoenix_fastdir/experiments/opticalflow/MOT17/LDOF/MOT17-02/bw'
    fw_flow_dir='/mnt/phoenix_fastdir/experiments/opticalflow/MOT17/LDOF/MOT17-02/fw'
    batch_rename_func(fw_flow_dir)

if __name__ == '__main__':
    mot_switch=False
    if mot_switch:
        motdevkit_16_to_17()
    
    seg_label_dir='/mnt/phoenix_fastdir/experiments/detection/MOT17/MRCNN/CheckProposal/Hungarian/BinPerPerson/MOT17-02'
    seg_switch=True
    
    labels=os.listdir(seg_label_dir)

    for label in labels:
        s_label_dir=os.path.join(seg_label_dir,label)
        batch_rename_func(s_label_dir)











    
    


