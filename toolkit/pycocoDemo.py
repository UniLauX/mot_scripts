##matplotlib inline
from pycocotools.coco import COCO
import numpy as np
import os
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2

pylab.rcParams['figure.figsize'] = (8.0, 10.0)


print('==========================testing pycocotools====================================')
rootDir= os.getcwd()

dataDir=os.path.join(rootDir,'data','coco')
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

print (annFile)

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))
print('cats num:', len(cats))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))
# print 'cats num:', len(cats)

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person','dog','skateboard'])
imgIds = coco.getImgIds(catIds=catIds )

print 'catIds:', catIds


imgIds = coco.getImgIds(imgIds = [324158])
##imgIds = coco.getImgIds(imgIds = [279278])
print 'imgIds:', imgIds

img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

##print img
# load and display image
# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image
I = io.imread(img['coco_url'])

plt.axis('off')
plt.imshow(I)
plt.show()

# rgb=I
# bgr = rgb[...,::-1]        ##matplotlib to OpenCV  
# ##rgb = bgr[...,::-1]      ##OpenCv to matplotlib
# cv2.imshow('im',bgr)
# cv2.waitKey(-1)

plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)

anns = coco.loadAnns(annIds)
                              
# for ann in anns:               ## annotation to masks
#     mask=coco.annToMask(ann)
#     ann1=coco.imgToAnns

coco.showAnns(anns)
plt.show()


annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
coco_kps=COCO(annFile)

# load and display keypoints annotations
plt.imshow(I); plt.axis('off')
ax = plt.gca()
annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco_kps.loadAnns(annIds)
coco_kps.showAnns(anns)
plt.show()


# initialize COCO api for caption annotations
annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco_caps=COCO(annFile)

# load and display caption annotations
annIds = coco_caps.getAnnIds(imgIds=img['id'])
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)
plt.imshow(I); plt.axis('off'); plt.show()
