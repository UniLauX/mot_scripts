import _init_paths
from mot_config import mcfg
import os

from common_lib import create_dir
from common_lib import create_recur_dirs
##===========================================input================================================
data_dir=mcfg.DATA.DATA_DIR
imgset=mcfg.DATA.IMGSET
jpgdir=mcfg.DATA.JPGDIR 
prop_res_dir=mcfg.PROPOSAL.DET_DIR    ## detlets need to be continious  
bin_mask_dir=mcfg.MASK.BIN_LAB_DIR    ## seglets need to be continious
optflow_dir=mcfg.OPTFLOW.BIN_FLOW_DIR
seg_res_dir=mcfg.MASK.RES_DIR
track_mask_folder='track_mask'


##(1) color images
def get_ims_info():
    imgset_path=os.path.join(data_dir,imgset) 
    jpgdir_path=os.path.join(data_dir,jpgdir)
    im_ext=mcfg.DATA.IMGEXT
    return imgset_path,jpgdir_path,im_ext

##(2) proposals
def get_dets_info(set_name):   
    prop_res_ext=mcfg.PROPOSAL.FILE_EXT    ## stored in .txt
    prop_file_path=os.path.join(prop_res_dir,set_name+prop_res_ext)
    return prop_file_path,prop_res_ext

##(3) binary mask per person
def get_segs_info(set_name):
    mask_dir=os.path.join(bin_mask_dir,set_name)
    mask_im_ext=mcfg.MASK.BIN_IM_EXT    #stored in '.sm'
    return mask_dir, mask_im_ext

##(4) optical flow(in the whole image)
def get_flows_info(set_name):
    flow_dir=os.path.join(optflow_dir,set_name)
    flow_im_ext=mcfg.OPTFLOW.FLOW_IMG_EXT   ##stored in '.flo'
    return flow_dir,flow_im_ext

##========================================== output ================================================
##(5) results directory:(in phoenix)
def get_track_mask_info(set_name):
    track_mask_dir=os.path.join(seg_res_dir,track_mask_folder)
    ##print 'seg_res_dir:', seg_res_dir
    track_mask_dir=os.path.join(track_mask_dir,set_name)
    create_recur_dirs(track_mask_dir)
    return track_mask_dir
##========================================== output ================================================
##(5) results directory:(in phoenix)
def get_vis_dir_info(vis_folder):
    track_mask_dir=os.path.join(seg_res_dir,track_mask_folder) 
    vis_mask_dir=os.path.join(track_mask_dir,vis_folder) 
    create_dir(vis_mask_dir) 
    return vis_mask_dir


   
'''
print '=========================input dir & path======================================='
    print 'imgset_path:',imgset_path
    print 'jpgdir_path:',jpgdir_path
    print 'im_ext:',im_ext
    print 'proposal_res_dir:',prop_res_dir
    print 'label mask dir:', bin_lab_dir
    print 'mask_im_ext:',mask_im_ext
    print 'flow dir:', bin_flow_dir
    print 'flow_im_ext:',flow_im_ext

    print '==================================output dir==============================================='
    print 'track_mask_dir:',track_mask_dir
    print '-----------------------------------------------------------------'
    '''