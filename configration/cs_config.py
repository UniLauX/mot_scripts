## define the configration for cityscapes dataset
from easydict import EasyDict as edict
import os

__C = edict()

# Consumers can get config by:
#   from mot_config import mcfg
cscfg = __C

##=========================================Data info=====================================
__C.DATA = edict()
__C.DATA.ROOT_DIR='/mnt/phoenix_fastdir/dataset/cityscapes/leftImg8bit_sequence'
__C.DATA.IMGSET_DIR=os.path.join(__C.DATA.ROOT_DIR,'imgset')
__C.DATA.BRANCH='val'
__C.DATA.IM_EXT = '.png'

##========================================detection==================================
__C.DETCTION = edict()

__C.DETCTION.RES_DIR='/mnt/phoenix_fastdir/experiments/detection/cityscapes'   ##result directory

__C.DETCTION.ALG='tf_m5' # Tensorflow/object_detection
##__C.DETCTION.ALG='FRCNN'
##__C.DETCTION.ALG='MRCNN'

##==========================================Demo====================================
__C.DEMO = edict()
__C.DEMO.VIDEO_FPS=30
