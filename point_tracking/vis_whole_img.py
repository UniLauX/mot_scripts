from mot_config import mcfg
import os
import numpy as np
import scipy.io as sio
import cv2


##note: Visualized in the whole image
##====================general functions================
## drawing with different color
def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def create_dir(dirpath):
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath) 



if __name__ == '__main__':

    print 'visualized point tracking results start...'
    ##(1) load color images
     
    data_dir=mcfg.PTRACK.DATA_DIR
    imgset=mcfg.PTRACK.IMGSET
    jpgdir=mcfg.PTRACK.JPGDIR  
    im_ext=mcfg.PTRACK.IMGEXT

    
    track_res_dir=mcfg.PTRACK.RES_DIR
    frame_interval=mcfg.PTRACK.FRAME_INTER
    sample_rate=mcfg.PTRACK.SAMPLE_RATE


    imgset_path=os.path.join(data_dir,imgset)
    jpgdir_path=os.path.join(data_dir,jpgdir)
    print imgset_path
    print jpgdir_path

    with open(imgset_path) as f:
        im_names = [x.strip()+im_ext for x in f.readlines()]
    f.close

    ##for testing: just 19 images included
    im_names=im_names[:frame_interval]

  
    sub_dir_name=str(frame_interval)+'frm_'+str(sample_rate)+'pt'

    ##(2) load point matrix
    point_mat_path= os.path.join(track_res_dir,sub_dir_name,'pt_mat.mat')
    
    ##(3) load label matrix
    label_mat_path= os.path.join(track_res_dir,sub_dir_name,'lab_mat.mat')
     
    track_res_dir=os.path.join(track_res_dir,sub_dir_name,'tracking') 
    create_dir(track_res_dir) 



    point_mat=sio.loadmat(point_mat_path)['X']      #point_mat.shape=(2F,D) ---> F is the framenumber and D is the (feature) point number

    label_mat=sio.loadmat(label_mat_path)['grp']    #label_mat.shape=(D,1)  ---> D is the(feature) point number
    
    frame_number=point_mat.shape[0]/2
    point_number=point_mat.shape[1]

    print 'point_mat.shape:', point_mat.shape
    print 'label_mat.shape:', label_mat.shape

     
    cmap = color_map()  ## color map
    
    for frm_id in xrange(frame_number):  #should be in frame_number
        im_path=os.path.join(data_dir,jpgdir,im_names[frm_id])
        rgb_im=cv2.imread(im_path)
        
        res_im_path=os.path.join(track_res_dir,im_names[frm_id])

        im_width=rgb_im.shape[1]
        im_height=rgb_im.shape[0]
        
        for pt_id in xrange(point_number): ##point_number
            x=int(np.round(point_mat[frm_id*2,pt_id]))
            y=int(np.round(point_mat[frm_id*2+1,pt_id]))
             
            x1=max(0,x-2)
            y1=max(0,y-2)
            x2=min(x+2,im_width-1)
            y2=min(y+2,im_height-1)

            label=int(label_mat[pt_id])
            color=cmap[label]
            colo= np.array((int(color[0]),int(color[1]),int(color[2])))
            
            cv2.rectangle(rgb_im,(int(x1),int(y1)),(int(x2),int(y2)),(colo),2)

        cv2.imwrite(res_im_path,rgb_im)
        ##cv2.imshow('rgb_im',rgb_im)
        ##cv2.waitKey(-1)
        
