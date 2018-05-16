import os
from common_lib import create_dir
 ##========================================iamge-net parameters setup===================================================================
upper_mot='MOT'
data_dir='/mnt/phoenix_fastdir/dataset'
jpgimgs_folder='JPEGImages'
imgsets_folder='ImageSets'
ppmimgs_folder='PPMImages'
annots_folder='Annotations'

imgsets_file_ext='.txt'
bmfsets_file_ext='.bmf'

def get_dataset_name(f_year):
    dataset_name=upper_mot+f_year
    return dataset_name

def get_imgnet_devkit_dir(f_year):
    imgnet_devkit_dir=os.path.join(data_dir,upper_mot+'devkit'+f_year)
    create_dir(imgnet_devkit_dir)
    return imgnet_devkit_dir

def get_data_dir(devkit_dir,dataset_name):
    data_dir=os.path.join(devkit_dir,dataset_name)
    create_dir(data_dir)
    return data_dir
     
def get_jpgimgs_dir(data_dir):
    jpgimgs_dir=os.path.join(data_dir,jpgimgs_folder)
    create_dir(jpgimgs_dir)
    return jpgimgs_dir

def get_annotations_dir(data_dir):
    annots_dir=os.path.join(data_dir,annots_folder)
    create_dir(annots_dir)
    return annots_dir

def get_imgsets_dir(data_dir):
    imgsets_dir=os.path.join(data_dir,imgsets_folder)
    create_dir(imgsets_dir)
    return imgsets_dir

def get_imgsets_path(imgsets_dir,set_prop):
    imgsets_main_dir=os.path.join(imgsets_dir,'Main')
    create_dir(imgsets_main_dir)
    imgsets_path=os.path.join(imgsets_main_dir,set_prop+imgsets_file_ext)
    return imgsets_path

##-----------------------------------------------------------------------
def get_ppmimgs_dir(data_dir):
    ppmimgs_dir=os.path.join(data_dir,ppmimgs_folder)
    create_dir(ppmimgs_dir)
    return ppmimgs_dir

def get_bmfsets_path(imgsets_dir,set_name):
    imgsets_main_dir=os.path.join(imgsets_dir,'Bmf')
    create_dir(imgsets_main_dir)
    bmfsets_path=os.path.join(imgsets_main_dir,set_name+bmfsets_file_ext)
    return bmfsets_path