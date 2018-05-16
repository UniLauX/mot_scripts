## configration file

# `pip install easydict` if you don't have it
from easydict import EasyDict as edict
import os

__C = edict()
# Consumers can get config by:
#   from mot_config import mcfg
mcfg = __C

upper_dataname='MOT'
s_year='15'
f_year='2015'
set_pro='test'
seq_name='TUD-Crossing' ## need to adjust to suitable for mult seq
iter_num=5 ## org=5
gt_start_index=1
seq_frm_num=201   ## org=20
shift_num=0     ##
devkit_folder=upper_dataname+'devkit'+f_year   # eg. MOTdevkit2017
s_dataset=upper_dataname+s_year
f_dataset=upper_dataname+f_year

##=========================================Img info=====================================
__C.DATA = edict()
__C.DATA.DATA_DIR='/mnt/phoenix_fastdir/dataset'
__C.DATA.IMGEXT='.jpg'

__C.DATA.IMGSET=os.path.join(devkit_folder,f_dataset,'ImageSets','Main',set_pro+'.txt')    ##eg.'MOTdevkit2017/MOT2017/ImageSets/Main/val.txt' 
##__C.DATA.IMGSET_TRAIN='MOTdevkit2017/MOT2017/ImageSets/Main/train.txt'
__C.DATA.JPGDIR=os.path.join(devkit_folder,f_dataset,'JPEGImages')   ##eg.'MOTdevkit2017/MOT2017/JPEGImages'
__C.DATA.DATASET=s_dataset
__C.DATA.SEQNAME=seq_name
__C.DATA.SEQ_GT_START_INDEX=gt_start_index
__C.DATA.SEQ_LENGTH=seq_frm_num
__C.DATA.SEQ_SHIFT=shift_num

##======================================= experiments==================================
__C.EXPR = edict()
__C.EXPR.EXPR_DIR='/mnt/phoenix_fastdir/experiments'
__C.EXPR.DET_DIR=os.path.join(__C.EXPR.EXPR_DIR,'detection')
__C.EXPR.SEG_DIR=os.path.join(__C.EXPR.EXPR_DIR,'segmentation')
__C.EXPR.FLOW_DIR=os.path.join(__C.EXPR.EXPR_DIR,'opticalflow')
__C.EXPR.DET_ALGR='MRCNN'
__C.EXPR.SEG_ALGR='MRCNN'

##=======================================proposals=========================================
__C.PROPOSAL = edict()
## folders
__C.PROPOSAL.DET_FOLDER='det_proposals'
__C.PROPOSAL.SEG_FOLDER='seg_proposals'

##__C.PROPOSAL.RES_DIR= os.path.join(__C.EXPR.DET_DIR,__C.DATA.DATASET,__C.EXPR.DET_ALGR,'CheckProposal')
__C.PROPOSAL.RES_DIR= os.path.join(__C.EXPR.DET_DIR,__C.DATA.DATASET,__C.EXPR.DET_ALGR,'CheckProposalContinious')

__C.PROPOSAL.ALGN_ALGR='Hungarian'
__C.PROPOSAL.RES_DIR=os.path.join(__C.PROPOSAL.RES_DIR,__C.PROPOSAL.ALGN_ALGR)
__C.PROPOSAL.DET_DIR=os.path.join(__C.PROPOSAL.RES_DIR,__C.PROPOSAL.DET_FOLDER) 
__C.PROPOSAL.FILE_EXT='.txt'

##========================================= mask =================================================
__C.MASK = edict()
##__C.MASK.BIN_LAB_DIR='/home/uni/Lab/Experiments/mask-rcnn/CheckProposal/MOT16/Hungarian/BinPerPerson'

#__C.MASK.BIN_LAB_DIR=os.path.join(__C.PROPOSAL.RES_DIR,'BinPerPerson') #discrete
__C.MASK.BIN_LAB_DIR=os.path.join(__C.PROPOSAL.RES_DIR,__C.PROPOSAL.SEG_FOLDER)   
__C.MASK.BIN_IM_EXT='.sm'
__C.MASK.RES_DIR= os.path.join(__C.EXPR.SEG_DIR,__C.DATA.DATASET,__C.EXPR.SEG_ALGR)

##======================================= optical flow==============================================
__C.OPTFLOW = edict()
##__C.OPTFLOW.BIN_FLOW_DIR='/home/uni/Lab/Experiments/mask-rcnn/OpticalFlow/MOT16'
__C.OPTFLOW.BIN_FLOW_DIR=os.path.join(__C.EXPR.FLOW_DIR,__C.DATA.DATASET)

__C.OPTFLOW.FLOWNAME='LDOF'
__C.OPTFLOW.FLOW_IMG_EXT='.flo'
__C.OPTFLOW.TYPE_LDOF=True  ## default
__C.OPTFLOW.TYPE_PYFLOW=False
__C.OPTFLOW.TYPE_DEEPFLOW=False

if __C.OPTFLOW.TYPE_LDOF:
    __C.OPTFLOW.FLOWNAME='LDOF'
    __C.OPTFLOW.FLOW_IMG_EXT='.flo'

if __C.OPTFLOW.TYPE_PYFLOW:
      __C.OPTFLOW.FLOWNAME='pyflow'

if __C.OPTFLOW.TYPE_DEEPFLOW:
      __C.OPTFLOW.FLOWNAME='DeepFlow'

__C.OPTFLOW.BIN_FLOW_DIR=os.path.join(__C.OPTFLOW.BIN_FLOW_DIR,__C.OPTFLOW.FLOWNAME)

##=========================================flow tracking============================================
__C.FTRACK = edict()
__C.FTRACK.ITER_NUM=iter_num ## default

##XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX data transfer to phoenix xxxxxxxxxxxxxxxxxxxxxxxxxx##
##========================================detection==================================
__C.DETCTION = edict()

__C.DETCTION.RES_DIR='/mnt/phoenix_fastdir/experiments/detection/MOT'   ##result directory

__C.DETCTION.ALG='tf_m5' # Tensorflow/object_detection

##=========================================point tracking=====================================
__C.PTRACK = edict()

## point tracking
__C.PTRACK.RES_DIR='/home/uni/Lab/Experiments/mask-rcnn/SubspaceClustering/PointTracking'

__C.PTRACK.FRAME_INTER=10  #how many frame that EDSC(subspace clustering conduct)

__C.PTRACK.SAMPLE_RATE=8   #point sampling rate

__C.PTRACK.OBJECT_SIZE_THRESHOLD=150

##========================================segmentation==================================
#__C.SEGMENTATION = edict()

#__C.SEGMENTATION.RES_DIR='/mnt/phoenix_fastdir/experiments/segmentation/'   ##result directory

#__C.SEGMENTATION.ALG='MRCNN' # Tensorflow/object_detection

##==========================================Demo====================================
__C.DEMO = edict()
__C.DEMO.VIDEO_FPS=30

##=======================================visualization=========================================
__C.VISUAL = edict()
__C.VISUAL.RES_DIR='/home/uni/Lab/Experiments/mask-rcnn/VisComp'
__C.VISUAL.TRACK_MASK_DIR=os.path.join(__C.VISUAL.RES_DIR,'VisTrackMask')



