import _init_paths
import os
import mot_dev as modv
import imgnet_dev as imdv

from mot_util import load_seq_names
from mot_util import generate_imgsets
from mot_util import generate_jpgimgs 
from mot_util import generate_ppmimgs
from mot_util import generate_bmfsets

debug_flag=True
gene_jpgimgs_flag=False
gene_imgsets_flag=False
gene_annots_flag=False

gene_ppms_flag=False
gene_bmf_flag=False

## read from mot_devkit info
def get_mot_devkit_info(s_year,set_pro):
    s_dataset_name=modv.get_dataset_name(s_year)
    seq_map_name=modv.get_seq_map_name(s_year,set_pro)
    seqs_data_dir=modv.get_seqs_data_dir(s_dataset_name,set_pro)

    seq_map_path=modv.get_seq_map_path(seq_map_name)
    seq_names=load_seq_names(seq_map_path)
    
    return seqs_data_dir,seq_names

def get_imgnet_devkit_info(f_year,set_pro):

    f_dataset_name=imdv.get_dataset_name(f_year)
    imgnet_devkit_dir=imdv.get_imgnet_devkit_dir(f_year)
    imgnet_data_dir=imdv.get_data_dir(imgnet_devkit_dir,f_dataset_name)
    
    imgnet_jpgimgs_dir=imdv.get_jpgimgs_dir(imgnet_data_dir)
    imgnet_annots_dir=imdv.get_annotations_dir(imgnet_data_dir)
    imgnet_imgsets_dir=imdv.get_imgsets_dir(imgnet_data_dir)
    imgnet_imgsets_path=imdv.get_imgsets_path(imgnet_imgsets_dir,set_pro)
  
    imgnet_ppmimgs_dir=imdv.get_ppmimgs_dir(imgnet_data_dir)
    return imgnet_jpgimgs_dir,imgnet_annots_dir,imgnet_imgsets_path,imgnet_ppmimgs_dir,imgnet_imgsets_dir



if __name__ == '__main__':

    #-----------------------------------------specific parameters--------------------------------------------------------------------------------------------
    year_id=0    ## 0:2015, 1:2016, 2:2017
    set_pro_id=2 ## 0:train 1:val  2:test
  
    s_year=modv.get_short_year(year_id)
    f_year=modv.get_full_year(year_id)
    set_pro=modv.get_set_property(set_pro_id)

    ## mot devkit
    seqs_data_dir,seq_names=get_mot_devkit_info(s_year,set_pro)
    
    ## imgnet devkit
    imgnet_jpgimgs_dir,imgnet_annots_dir,imgnet_imgsets_path, imgnet_ppmimgs_dir,imgnet_imgsets_dir=get_imgnet_devkit_info(f_year,set_pro)

    if debug_flag:
        print 'seqs_data_dir:',seqs_data_dir
        print 'seq_names:',seq_names
        print 'imgnet_jpgimgs_dir:',imgnet_jpgimgs_dir
        print 'imgnet_annots_dir:',imgnet_annots_dir
        print 'imgnet_imgsets_path:',imgnet_imgsets_path  
   
    ##(1) rename and copy images:
    if gene_jpgimgs_flag:
        generate_jpgimgs(seqs_data_dir,seq_names,imgnet_jpgimgs_dir)
          
    if gene_imgsets_flag:
        generate_imgsets(seqs_data_dir,seq_names,imgnet_imgsets_path)
   
    if gene_ppms_flag:
        generate_ppmimgs(seqs_data_dir,seq_names,imgnet_ppmimgs_dir)

    if gene_bmf_flag:
        generate_bmfsets(seqs_data_dir,seq_names,imgnet_imgsets_dir)
    
  




   
