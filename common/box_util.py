import numpy as np

# # x1,y1,w,h -> x1,y1,x2,y2
# def coord_to_bbox(coord):
#     x1=coord[0]
#     y1=coord[1]
#     w=coord[2]
#     h=coord[3]
#     x2=x1+w-1
#     y2=y1+h-1
#     bbox=[x1,y1,x2,y2]
#     return bbox

##--------------------------------------- boxes----------------------------------------------------------
##(x1,y1,w,h) to (x1,y1,x2,y2)
def coords_to_boxes(coords):
    boxes=np.zeros(coords.shape)
    boxes[:,:2]=coords[:,:2]  ## x1=x1, y1=y1
    boxes[:,2]=coords[:,0]+coords[:,2]-1  ## x2=x1+w-1
    boxes[:,3]=coords[:,1]+coords[:,3]-1  ## y2=y1+h-1
    return boxes

##(x1,y1,x2,y2) to (x1,y1,w,h)
def boxes_to_coords(boxes):
    coords=np.zeros(boxes.shape)
    coords[:,:2]=boxes[:,:2]  ## x1=x1, y1=y1
    coords[:,2]=boxes[:,2]-boxes[:,0]+1  ## x2=x1+w-1
    coords[:,3]=boxes[:,3]-boxes[:,1]+1  ## y2=y1+h-1
    return coords

## based on mask's(in masks) shape in 
def boxes_align_masks(masks,in_boxes):
    if len(masks)!=len(in_boxes):
        print 'mask_num is not equal to box_num...'
    out_boxes=in_boxes
    box_num=len(in_boxes)
    for n_id in xrange(box_num):
            i_box=in_boxes[n_id]
            mask=masks[n_id]
            x1=max(0,int(np.round(i_box[0])))
            y1=max(0,int(np.round(i_box[1])))
            w=mask.shape[1]
            h=mask.shape[0]
            x2=x1+w-1
            y2=y1+h-1            
            o_box=[x1,y1,x2,y2]
            out_boxes[n_id]=o_box
    return out_boxes

## get tight_box from binray mask
def get_mini_box_from_mask(binarymask):
    one_pos=np.where(binarymask>=1)
    arr_pos=np.asarray(one_pos)
    h_inds=arr_pos[0,:]  #h  ## due to the binarymask(encode in h,w order)
    w_inds=arr_pos[1,:]  #w 

    if len(w_inds)==0:  ## special case, for empty mask
        box=[0,1,0,1]
    else:
        x_min=min(w_inds)
        x_max=max(w_inds)
        y_min=min(h_inds)
        y_max=max(h_inds)
        w=x_max-x_min+1
        h=y_max-y_min+1
        box=[x_min, y_min, x_max, y_max]
    return box

def get_int_box_xyxy(bbox,im_shape):
    im_width=im_shape[1]
    im_height=im_shape[0]
    x1 = int(round(bbox[0]))
    y1 = int(round(bbox[1]))
    x2= int(round(bbox[2]))
    y2= int(round(bbox[3]))
    x1 = np.min((im_width - 1, np.max((0, x1))))
    y1 = np.min((im_height - 1, np.max((0, y1))))
    x2 = np.min((im_width - 1, np.max((0, x2))))
    y2 = np.min((im_height - 1, np.max((0, y2))))
    return x1,y1,x2,y2
