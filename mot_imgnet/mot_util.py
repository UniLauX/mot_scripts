import _init_paths
from common_lib import create_dir
from common_lib import load_txt_to_strarr
from common_lib import save_strarr_to_txt
from trans_util import rename_copy_images
from trans_util import rename_create_ppmimgs
import numpy as np
import os

debug_flag=True

def load_seq_names(seqmap_path):
    str_arr=load_txt_to_strarr(seqmap_path)
    seq_names=str_arr[1:]
    return seq_names

def generate_jpgimgs(seq_data_dir,seq_names,dest_jpgimgs_dir):
    for seq_name in seq_names:
        seq_folder_dir=os.path.join(seq_data_dir,seq_name)
        seq_img1_dir=os.path.join(seq_folder_dir,'img1')
        rename_copy_images(seq_img1_dir,dest_jpgimgs_dir,seq_name)

def generate_ppmimgs(seq_data_dir,seq_names,dest_ppmimgs_dir):
    if not os.path.exists(dest_ppmimgs_dir):
        create_dir(dest_ppmimgs_dir)
    for seq_name in seq_names:
        seq_folder_dir=os.path.join(seq_data_dir,seq_name)
        seq_img1_dir=os.path.join(seq_folder_dir,'img1')
        rename_create_ppmimgs(seq_img1_dir,dest_ppmimgs_dir,seq_name)

def generate_imgsets(seqs_data_dir,seq_names,dest_imgsets_path):
    ##im_names(.jpg)
    str_arr=[]
    for seq_name in seq_names:
        seq_data_dir=os.path.join(seqs_data_dir,seq_name)
        seq_img1_dir=os.path.join(seq_data_dir,'img1')
        im_names=os.listdir(seq_img1_dir)
        im_names=np.sort(im_names)
        for im_name in im_names:
            base_im_name=os.path.splitext(im_name)[0]
            dest_im_name=seq_name+'_'+base_im_name
            str_arr.append(dest_im_name)
    dest_im_names=np.array(str_arr)
  
    ##save
    save_strarr_to_txt(dest_imgsets_path,dest_im_names)

##get annotations when necessary
def generate_annots():
    print 'generate_annots....'
    # if year==2015:
    #     opticn one
    # else:
    #     option two
   

def generate_bmfsets(seq_data_dir,seq_names,dest_imgsets_dir):
    dest_ppmsets_dir=os.path.join(dest_imgsets_dir,'Bmf')
    
    if debug_flag:
        print dest_ppmsets_dir

    ##im_names(.ppm)
    for seq_name in seq_names:
        seq_folder_dir=os.path.join(seq_data_dir,seq_name)
        seq_img1_dir=os.path.join(seq_folder_dir,'img1')
        im_names=os.listdir(seq_img1_dir)
        im_names=np.sort(im_names)
        ## file_path
        dest_bmffile_path=os.path.join(dest_ppmsets_dir,seq_name+'.bmf')
        ## im_names
        str_arr=[]
        first_line=str(len(im_names))+" "+'1'
        str_arr.append(first_line)

        for im_name in im_names:
            tmp_im_name=seq_name+'_'+im_name
            dest_im_name=tmp_im_name.replace('.jpg','.ppm')
            str_arr.append(dest_im_name)
            dest_im_names=np.array(str_arr)
            ##save
            ## add total_frame_num and view_count(1) at the very begining
            save_strarr_to_txt(dest_bmffile_path,dest_im_names)    

## need to concise the data format
## write det_arr in MOT-like format( with split_sign)
def write_detarr_as_mot(file_path,det_float_arr,split_sign=','):
    print '============================write detarr as mot===================================================='
    det_str_arr=det_float_arr.astype(str)
    col_num=det_str_arr.shape[1]

    with open(file_path, 'w') as f:
        for str_row in det_str_arr:
            tmp_row=str_row[0]
            for col_id in xrange(1,col_num):
                tmp_row=tmp_row+split_sign+str_row[col_id]     
            f.write('{:s}\n'. format(tmp_row))
    f.close()         




