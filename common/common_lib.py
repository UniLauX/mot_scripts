import os
import numpy as np
import cv2
# import cPickle
#load color images from local disk
def load_color_images(im_dir,im_names,im_ext):
    ims=[]
    for im_name in im_names:
        im_path=os.path.join(im_dir,im_name+im_ext)
        if not os.path.exists(im_path):
            print im_path,'not exist'
            break
        im=cv2.imread(im_path)
        ims.append(im)

    ##cv2.imshow('im',ims[0])
    ##cv2.waitKey(-1)
    print 'load', len(ims), 'color images done...'
    return ims

# read .txt file and save to an array
def load_txt_to_fltarr(file_path):
    arr=[]
    with open(file_path) as f:
        rows= [x.strip() for x in f.readlines()]
        for row in rows: 
            arr.append(row.split(','))
    arr = np.array(arr, dtype=np.float32)    #transfer list to np.array
    return arr


def save_strarr_to_txt(file_path,str_arr):
    with open(file_path, 'w') as f:
        for str_row in str_arr:
                f.write('{:s}\n'. format(str_row))
    f.close()             


# read .txt file and save to an array
def load_txt_to_strarr(file_path):
    arr=[]
    with open(file_path) as f:
        rows= [x.strip() for x in f.readlines()]
        for row in rows: 
            arr.append(row)
    arr = np.array(arr)    #transfer list to np.array
    return arr


# create directory
def create_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path) 

## create a series of dirs
def create_recur_dirs(dirpath):
    num=len(dirpath.split('/'))-1
    dir_arr=[]
    while num>0:
        if not os.path.exists(dirpath):
            dir_arr.append(dirpath)
            parent_dir=os.path.dirname(dirpath)
            dirpath=parent_dir
            num=num-1 
        else:
            num=num-1

    c_len=len(dir_arr)
    for c_id in xrange(c_len,0,-1):
        create_dir(dir_arr[c_id-1])

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

# ##------------------------------------ mask util --------------------------------------------------
# ## write binary mask into file_path (.sm format)
# def save_binary_mask(file_path, bin_mask):
#     with open(file_path, 'wb') as f_save:
#         cPickle.dump(bin_mask, f_save, cPickle.HIGHEST_PROTOCOL)

# ## load  binary mask from file_path
# def load_binary_mask(file_path):
#     with open(file_path, 'rb') as f:
#         bin_mask = cPickle.load(f)
#         return bin_mask

##--------------------------------------common lib ----------------------------------------------------
## get tight box from binary mask
def get_mini_box_from_mask(binarymask):
    box=np.zeros((4,))
    one_pos=np.where(binarymask>=1)
    arr_pos=np.asarray(one_pos)

    h_inds=arr_pos[0,:]  #h  ## due to the binarymask(encode in h,w order)
    w_inds=arr_pos[1,:]  #w 

    x_min=min(w_inds)
    x_max=max(w_inds)
    y_min=min(h_inds)
    y_max=max(h_inds)
    w=x_max-x_min+1
    h=y_max-y_min+1
    box[:]=x_min, y_min, x_max, y_max
    return box   # type is array (4,)