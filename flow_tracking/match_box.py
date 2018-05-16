import _init_paths
import cv2
import os
import numpy as np
from scipy import ndimage
# from match_box import color_diff_matching as bf_match_box
# from match_box import match_template as mt_match_box

##from circulant_matrix_tracker import track_bbox as cm_tracker 
##other methods for tracker
##cm_tracker(im_dir)
#cv2.calcOpticalFlowFarneback()

## matching template link
##https://docs.opencv.org/trunk/d4/dc6/tutorial_py_template_matching.html

##https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.filters.convolve.htmld


move_thre=10.0  ## threshold of max shift distance

## get box-center shift based on patch matching(of color image)
def correct_shift_vec(in_vec):
    vec_num=len(in_vec)
    
    for v_id in xrange(vec_num):
        i_vec=in_vec[v_id]
        
        if v_id==0:   ## first frame
            if i_vec[0]>move_thre or i_vec[0]<-move_thre or i_vec[1]>move_thre or i_vec[1]<-move_thre:
                in_vec[v_id]=in_vec[v_id+1]
        elif v_id==vec_num-1:   ## last frame
            if i_vec[0]>move_thre or i_vec[0]<-move_thre or i_vec[1]>move_thre or i_vec[1]<-move_thre:
                in_vec[v_id]=in_vec[v_id-1]
        else: 
            if i_vec[0]>move_thre or i_vec[0]<-move_thre or i_vec[1]>move_thre or i_vec[1]<-move_thre:
                in_vec[v_id]=(in_vec[v_id-1]+in_vec[v_id+1])/2.0

    return in_vec



def get_bboxes_center_shift(ims,bboxes,motion_estm_meth='match_template'):
    im_num=len(ims)
    box_num=len(bboxes)
    
    if im_num!=box_num:
        print 'image number is not equal to box number...'
        return
    if im_num<2:
        print 'image number is less than 2, could not do bboxes_center_shift...'
        return   
    vec_len=box_num-1
    shift_vec=np.zeros((vec_len,2),dtype=float)  ##(x01,y01),(x12,y12),(x23,y23)
    
    if motion_estm_meth=='match_template':
        print motion_estm_meth,'...'
        for s_id in xrange(vec_len):
             swap_flag, map_bbox,bbox_center_shift=match_template(ims[s_id],ims[s_id+1],bboxes[s_id],bboxes[s_id+1])
             shift_vec[s_id]=bbox_center_shift

    elif motion_estm_meth=='bruteforce':
        print 'bruteforce...'
        for s_id in xrange(vec_len):
             swap_flag, map_bbox,bbox_center_shift=color_diff_matching(ims[s_id],ims[s_id+1],bboxes[s_id],bboxes[s_id+1])
             shift_vec[s_id]=bbox_center_shift
    else:## offline by bruthforce
        print 'no motion considering...'
          #shift_vec=[[ 4. , 0.],[ 5. , 0.],[ 5. , 0.],[ 5. , 1.],[ 6. , 0.],[ 5. , 1.],[ 6. , 0.],[ 7. , 0.],[ 8. , 0.],[ 7. , 0.],[ 9. , 0.],[ 8. , 0.],[ 9. , 0.],[ 8. , 0.],[ 8. , 0.],[ 8. , 0.],[ 9. , 0.],[ 8. , 0.],[ 8. , 0.]]  
    return shift_vec


def conv_matching():
    ##=====================kernel from first image=======================================
     ##if(im1_w>100)     
     ## odd numbers:  in middle
     ## even numbers: in latter
     s_ratio=2  
    #  k_cx=im1_x1+im1_w/2  
    #  k_w=im1_w/s_ratio
    #  k_x1=max(0, k_cx-k_w/2)
    #  k_x2=min(im_width-1,k_cx+(k_w-1)/2)
     
    #  k_cy=im1_y1+im1_h/2
    #  k_h=im1_h/s_ratio
    #  k_y1=max(0, k_cy-k_h/2)
    #  k_y2=min(im_height-1,k_cy+(k_h-1)/2)
     
    # #  print 'k_cx:', k_cx
    # #  print 'im1_w:',im1_w
    # #  print 'k_w:', k_w
    # #  print 'k_x1:',k_x1
    # #  print 'k_x2:',k_x2

    # ##===========================search space from second image======================================================
    #  a_x1=x_min
    #  a_y1=y_min
    #  a_x2=x_max
    #  a_y2=y_max
    #  a_w=a_x2-a_x1+1
    #  a_h=a_y2-a_y1+1
    
    #  ##====================================== convole calculating==================================================
    #  conv_sum=np.zeros((a_h,a_w))
    #  for c_id in xrange(2):              # r,g,b three channels
    #      a=im2[a_y1:a_y2 + 1, a_x1:a_x2 + 1,c_id]
    #  ##k=im1[im1_y1:im1_y2+1,im1_x1:im1_x2+1,0]
    #      k=im1[k_y1:k_y2+1,k_x1:k_x2+1,c_id]
    #      conv_out=ndimage.convolve(a, k, mode='constant', cval=0.0)
    #      conv_sum=conv_sum+conv_out
    #      ##print conv_out.shape
    #      #print conv_out[0,0]
    #  max_val=np.max(conv_sum)
    #  max_idx=np.where(conv_sum==max_val)

    #  m_cy=max_idx[0][0]  ## y_center
    #  m_cx=max_idx[1][0]  ## x_center

    #  ## new kernel box
    #  new_cy=a_y1+m_cy
    #  new_cx=a_x1+m_cx
    #  n_k_x1=max(0, new_cx-k_w/2)
    #  n_k_x2=min(im_width-1,new_cx+(k_w-1)/2)
    #  n_k_y1=max(0, new_cy-k_h/2)
    #  n_k_y2=min(im_height-1,new_cy+(k_h-1)/2) 
     
    #  #  ##rect
    #  cv2.rectangle(im1,(im1_x1,im1_y1),(im1_x2,im1_y2),(255,0,0),2) 
    #  cv2.rectangle(im1,(k_x1,k_y1),(k_x2,k_y2),(255,0,0),2)
    #  cv2.rectangle(im2,(im2_x1,im2_y1),(im2_x2,im2_y2),(0,155,0),2) 
    #  #  cv2.rectangle(im2,(im1_x1,im1_y1),(im1_x2,im1_y2),(255,0,0),2) 
    #  cv2.rectangle(im2,(n_k_x1,n_k_y1),(n_k_x2,n_k_y2),(0,255,0),2)
    #  cv2.rectangle(im2,(k_x1,k_y1),(k_x2,k_y2),(255,0,0),2)

    #  cv2.rectangle(im1,(x_min,y_min),(x_max,y_max),(0,0,255),2) 
    #  cv2.rectangle(im2,(x_min,y_min),(x_max,y_max),(0,0,255),2) 
    #  cv2.imshow('im1',im1)
    #  cv2.imshow('im2',im2)
    #  cv2.waitKey(-1)
     
    #  ## 
    #  #511frm:  1481.12,380.06,228.341,667.302
    #  ##512frm: 1469.27,398.2,173.183,536.783,
    #  ##513frm: 1466.7,420.843,169.237,498.060
    #  ##514,1479.13,415.131,164.700,455.671,
    #  ##515, 1484.92,412.434,210.074,515.402
    #  ##print "Matchming Boxes..."

def get_rect_area(rect):
    rect_area=0
    r_w=rect[2]-rect[0]+1
    r_h=rect[3]-rect[1]+1
    rect_area=r_w*r_h
    return rect_area

def get_color_diff(im1,im2,rect1,rect2):
    color_diff=0.0
        
    rect_w=rect1[2]-rect1[0]+1
    rect_h=rect1[3]-rect1[1]+1 
    
    ##rect_im1=np.zeros((rect_h,rect_w,3),dtype=float)
    ##rect_im2=np.zeros((rect_h,rect_w,3),dtype=float)

    rect_im1=im1[rect1[1]:rect1[3]+1,rect1[0]:rect1[2]+1,:]
    rect_im2=im2[rect2[1]:rect2[3]+1,rect2[0]:rect2[2]+1,:]
    rect_diff=np.zeros((rect_h,rect_w),dtype=float)
    ##diff_img=np.absolute(rect_im1-rect_im2)
    ##r_im1=rect_im1.copy()
    ##r_im2=rect_im2.copy()

    # if rect_im1.shape != rect_im2.shape:
    #     print 'rect_im1 and rect_im2 are not equal size..'
    #     print 'rect_im1.shape:', rect_im1.shape
    #     print 'rect_im2.shape:', rect_im2.shape
    #     return
    #a=rect_im1*1.0-rect_im2*1.0
 
    ##print 'rect_im1.shape:',(rect_im1*1.0).shape
    ##print 'rect_im2.shape:', (rect_im2*1.0).shape
    #diff_img=np.absolute(rect_im1*1.0-rect_im2*1.0)

    #rect_diff=np.subtract(rect_im1,rect_im2)
    for c_id in xrange(3):
        rect_diff=rect_diff+np.absolute(rect_im1[:,:,c_id]*1.0-rect_im2[:,:,c_id]*1.0)
    
    rect_diff=rect_diff/3.0
    color_diff=np.mean(rect_diff)

    return color_diff

def color_diff_matching(im1,im2,bbox1,bbox2): 

    ## the smaller one is rotating in the larger one.
    ## by default, assume the bbox1 is the smaller one(exchange_flag=0)
    swap_flag=False
    rot_bbox=bbox1
    ref_bbox=bbox2 
    rot_img=im1
    ref_img=im2

    ## release to make the small one(rotate one) in the large one(reference one)##
    # bbox_area1=get_rect_area(bbox1)
    # bbox_area2=get_rect_area(bbox2)
    # if bbox_area1>bbox_area2:
    #     swap_flag=True
    #     rot_bbox=bbox2
    #     ref_bbox=bbox1
    #     rot_img=im2
    #     ref_img=im1

    im_width=ref_img.shape[1]
    im_height=ref_img.shape[0]

    rot_x1,rot_y1,rot_x2,rot_y2=rot_bbox
    rot_w=rot_x2-rot_x1+1  
    rot_h=rot_y2-rot_y1+1
    rot_cx=rot_x1+rot_w/2
    rot_cy=rot_y1+rot_h/2
    
    ref_x1,ref_y1,ref_x2,ref_y2=ref_bbox
    ref_w=ref_x2-ref_x1+1
    ref_h=ref_y2-ref_y1+1
   
    mean_color_diff=np.zeros((ref_h,ref_w))
   
    map_bbox=[0,0,0,0]
    ##bbox_center_shift=(0,0)

    for rf_cy in xrange(ref_y1,ref_y2+1):    ##ref_y1, ref_y2+1
        for rf_cx in xrange(ref_x1,ref_x2+1):  ## ref_x1, ref_x2+1
            map_x1=rf_cx-rot_w/2       ##left half
            map_x2=rf_cx+(rot_w-1)/2   ##right half
            rt_x2=rot_x2
            rt_x1=rot_x1

            if map_x1<0:                ## (1) beyond left edge
                out_val1=0-map_x1 
                rt_x1=rot_x1+out_val1
                map_x1=0
                # print 'map_x1:', map_x1
                # if rt_x1>im_width-1:
                #     print 'wrong number for rt_x1'
                #     return
            if map_x2>(im_width-1):       ## (2) beyond right edge
                out_val2=map_x2-(im_width-1)
                rt_x2=rot_x2-out_val2
                map_x2=im_width-1
                # print '--------------------------------------'
                # print 'map_x1:', map_x1
                # print 'map_x2:', map_x2
                # print 'm_w:', map_x2-map_x1+1
                # print 'rt_x1:', rt_x1
                # print 'rt_x2:', rt_x2
                # print 'r_w:',rt_x2-rt_x1+1
                # if rt_x2<0:
                #     print 'wrong number for rt_x2'
                #     return

            map_y1=rf_cy-rot_h/2       ##top half
            map_y2=rf_cy+(rot_h-1)/2   ##bottom half
            rt_y1=rot_y1
            rt_y2=rot_y2
            
            if map_y1<0:                ## (1) beyond left edge
                out_val3=0-map_y1
                rt_y1=rot_y1+out_val3
                map_y1=0
                # print 'map_y1:', map_y1
                # if rt_y1>im_height-1:
                #     print 'wrong number for rt_y1'
                #     return

            if map_y2>(im_height-1):       ## (2) beyond right edge
                out_val4=map_y2-(im_height-1) 
                rt_y2=rot_y2-out_val4
                map_y2=im_height-1
                # print 'map_y2:', map_y2
                # if rt_y2<0:
                #     print 'wrong number for rt_y2'
                #     return 

            ## bug in (rot_cx,rot_cy)->(1623,711)
          
            rot_rect=[rt_x1,rt_y1,rt_x2,rt_y2]
            ref_rect=[map_x1,map_y1,map_x2,map_y2]
            # print '===================================='
            # print 'rf_cx:', rf_cx
            # print 'rf_cy:', rf_cy
            # print 'rot_rect:',rot_rect
            # print 'ref_rect:', ref_rect    
            
            ##print rf_cx,rf_cy
            ##print 'rt_x2-rt_x1:', rt_x2-rt_x1, 'map_x2-mapx1:', map_x2-map_x1
            # if (rt_x2-rt_x1) != (map_x2-map_x1) or (rt_y2-rt_y1)!=(map_y2-map_y1):
            #      print 'rot_rect not equal to ref rect...'
            #      print 'rf_cx:', rf_cx, 'rf_cy:', rf_cy
            #      print 'rt_x2-rt_x1:', rt_x2-rt_x1+1, 'map_x2-mapx1:', map_x2-map_x1
            #      break
            mean_color_diff[rf_cy-ref_y1,rf_cx-ref_x1]=get_color_diff(rot_img,ref_img,rot_rect,ref_rect)
        
            # # testing
            # im_dir='/mnt/phoenix_fastdir/experiments/local_experiments/mask-rcnn/MaskIoU/min_color_difference/exp2'
            # im_path1=os.path.join(im_dir,'im1.jpg')
            # im_path2=os.path.join(im_dir,'im2.jpg')
            # im_path3=os.path.join(im_dir,'diff_img.jpg')

            # map_bbox_color=(0,0,255)
            # cv2.rectangle(im1,(rot_rect[0],rot_rect[1]),(rot_rect[2],rot_rect[3]),map_bbox_color,2)
            # cv2.rectangle(im2,(ref_rect[0],ref_rect[1]),(ref_rect[2],ref_rect[3]),map_bbox_color,2)
            # cv2.imshow('im1',im1)
            # cv2.imshow('im2',im2)
            # cv2.imwrite(im_path1,im1)
            # cv2.imwrite(im_path2,im2)
            # cv2.imwrite(im_path3,diff_img)
            # cv2.waitKey(-1)

    ## note tmp
    min_diff_val=np.min(mean_color_diff)
    min_diff_idx=np.where(mean_color_diff==min_diff_val)
   
    min_yid=min_diff_idx[0][0]+ref_y1   ## y_idx(in reference image coordinate)
    min_xid=min_diff_idx[1][0]+ref_x1   ## x_idx(in reference image coordinate)

    mp_x1=min_xid-rot_w/2       ##left half
    mp_x2=min_xid+(rot_w-1)/2   ##right half

    mp_y1=min_yid-rot_h/2       ##left half
    mp_y2=min_yid+(rot_h-1)/2   ##right half    
    map_bbox=[mp_x1,mp_y1,mp_x2,mp_y2]

    # print 'mean_color_diff:\n', mean_color_diff  
    # print 'min_diff_val:',min_diff_val
    # print 'min_diff_idx:', min_diff_idx
    # print 'min_yid:', min_yid
    # print 'min_xid:', min_xid
    # print 'rot_bbox:', rot_bbox
    # print 'map_bbox:', map_bbox
    bbox_center_shift=[min_xid-rot_cx,min_yid-rot_cy]
    print 'bbox_center_shift:', bbox_center_shift 
    return swap_flag, map_bbox,bbox_center_shift

def bruteforce_matching_demo():
      ##img file information
     im_dir='/home/uni/Lab/projects/Python/maskrcnn/data/MOTdevkit2016/MOT2016/JPEGImages'
     im_ext='.jpg'
     im_names=['MOT16-02_000512','MOT16-02_000513','MOT16-02_000515']

     ##save visualized results
     res_im_dir='/mnt/phoenix_fastdir/experiments/local_experiments/mask-rcnn/MaskIoU/min_color_difference' 
     ##read images
     im_path1=os.path.join(im_dir,im_names[0]+im_ext)
     im1=cv2.imread(im_path1)
     
     im_path2=os.path.join(im_dir,im_names[1]+im_ext)
     im2=cv2.imread(im_path2)

     im_width=im1.shape[1]
     im_height=im1.shape[0]
     boder_shift=20

     ## im_512
     im1_x1=int(1469.27)
     im1_y1=int(398.2)
     im1_w=int(173.183)
     im1_h=int(536.783)
     im1_x2=im1_x1+im1_w-1
     im1_y2=+im1_y1+im1_h-1
     
     ## im_515
     im2_x1=int(1484.92)
     im2_y1=int(412.434)
     im2_w=int(210)
     im2_h=int(515)
     im2_x2=im2_x1+im2_w-1
     im2_y2=im2_y1+im2_h-1
     
     ## max box(doesn't use here)
     x_min=max(0,(min(im1_x1,im2_x1)-boder_shift))
     y_min=max(0,(min(im1_y1,im2_y1)-boder_shift))
     x_max=min(im_width-1,(max(im1_x2,im2_x2)+boder_shift))
     y_max=min(im_height-1,(max(im1_y2,im2_y2)+boder_shift)) 
    
     bbox1=[im1_x1,im1_y1,im1_x2,im1_y2]
     bbox2=[im2_x1,im2_y1,im2_x2,im2_y2]

     swap_flag,map_bbox,bbox_center_shift=color_diff_matching(im1,im2,bbox1,bbox2)
     
    #  ##(3) visualization
    #  bbox_color1=(255,0,0)
    #  bbox_color2=(0,255,0)
    #  map_bbox_color=(0,0,255)

    #  cv2.rectangle(im1,(im1_x1,im1_y1),(im1_x2,im1_y2),bbox_color1,2) 
    #  cv2.rectangle(im2,(im2_x1,im2_y1),(im2_x2,im2_y2),bbox_color2,2) 
    #  if not swap_flag:
    #      cv2.rectangle(im2,(map_bbox[0],map_bbox[1]),(map_bbox[2],map_bbox[3]),map_bbox_color,2)
    #  if swap_flag:
    #     cv2.rectangle(im1,(map_bbox[0],map_bbox[1]),(map_bbox[2],map_bbox[3]),map_bbox_color,2)
    #  cv2.imshow('im1',im1)
    #  cv2.imshow('im2',im2)
    #  im_path1=os.path.join(res_im_dir,'im1.jpg')
    #  im_path2=os.path.join(res_im_dir,'im2.jpg')
    #  cv2.imwrite(im_path1,im1)
    #  cv2.imwrite(im_path2,im2)
    #  cv2.waitKey(-1)

def match_template(im1,im2,bbox1,bbox2):
    swap_flag=False
    ## color image to gray image
    g_im1=cv2.cvtColor( im1,cv2.COLOR_BGR2GRAY )
    g_im2=cv2.cvtColor( im2,cv2.COLOR_BGR2GRAY )  

    template=g_im1[bbox1[1]:bbox1[3]+1,bbox1[0]:bbox1[2]+1]
    w, h = template.shape[::-1]

    ## cv2.TM_CCORR (not robust)
    ## cv2.TM_SQDIFF(best one)
    methods = ['cv2.TM_SQDIFF','cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 
            'cv2.TM_CCORR_NORMED',  'cv2.TM_SQDIFF_NORMED']
    
    for meth in methods[:1]:
        img = g_im2.copy()
        method = eval(meth) 
        #print 'method:', method, ",", meth
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)    
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        m_x1=top_left[0]
        m_y1=top_left[1]
        m_x2=m_x1+w-1
        m_y2=m_y1+h-1
        bbox_center_shift=[m_x1-bbox1[0],m_y1-bbox1[1]] 
        ##print 'bbox_center_shift:', bbox_center_shift
        map_bbox=[m_x1,m_y1,m_x2,m_y2]
    return swap_flag,map_bbox,bbox_center_shift




if __name__ == '__main__':
    
     ##img file information
     im_dir='/home/uni/Lab/projects/Python/maskrcnn/data/MOTdevkit2016/MOT2016/JPEGImages'
     im_ext='.jpg'
     im_names=['MOT16-02_000512','MOT16-02_000513','MOT16-02_000515']
     ##save visualized results
     res_im_dir='/mnt/phoenix_fastdir/experiments/tracking/MOT17/match_bbox/match_template' 
     ##read images
     im_path1=os.path.join(im_dir,im_names[0]+im_ext)
     im1=cv2.imread(im_path1)  ##color image
     
     g_im1=cv2.imread(im_path1,0) ##gray image 

     im_path2=os.path.join(im_dir,im_names[1]+im_ext)
     im2=cv2.imread(im_path2)    ##color image 
     
     g_im2=cv2.imread(im_path2,0)  ##gray image

     im_width=im1.shape[1]
     im_height=im1.shape[0]
     boder_shift=20

     ## im_512
     ##im1_x1=int(1469.27)
     im1_x1=int(1747)
     im1_y1=int(398.2)
     im1_w=int(173.183)
     im1_h=int(536.783)
     im1_x2=im1_x1+im1_w-1
     im1_y2=+im1_y1+im1_h-1
     
     ## im_515
     im2_x1=int(1484.92)
     im2_y1=int(412.434)
     im2_w=int(210)
     im2_h=int(515)
     im2_x2=im2_x1+im2_w-1
     im2_y2=im2_y1+im2_h-1
     
     ## max box(doesn't use here)
     x_min=max(0,(min(im1_x1,im2_x1)-boder_shift))
     y_min=max(0,(min(im1_y1,im2_y1)-boder_shift))
     x_max=min(im_width-1,(max(im1_x2,im2_x2)+boder_shift))
     y_max=min(im_height-1,(max(im1_y2,im2_y2)+boder_shift)) 
    
     bbox1=[im1_x1,im1_y1,im1_x2,im1_y2]
     bbox2=[im2_x1,im2_y1,im2_x2,im2_y2]
     max_bbox=[x_min,y_min,x_max,y_max]
     ##swap_flag,map_bbox,bbox_center_shift=color_diff_matching(im1,im2,bbox1,bbox2)
     
    
    #  print im1
    #  cv2.imshow('im1',g_im1)
    #  cv2.waitKey(-1)

     swap_flag=False
     map_bbox,bbox_center_shift=match_template(im1,im2,bbox1,bbox2)

    #  ##(3) visualization
    #  bbox_color1=(0,0,255)
    #  bbox_color2=(0,255,0)
    #  map_bbox_color=bbox_color1

    #  cv2.rectangle(im1,(im1_x1,im1_y1),(im1_x2,im1_y2),bbox_color1,2) 
    #  cv2.rectangle(im2,(im2_x1,im2_y1),(im2_x2,im2_y2),bbox_color2,2) 

    #  print 'map_bbox:', map_bbox
    #  cv2.rectangle(im2,(map_bbox[0],map_bbox[1]),(map_bbox[2],map_bbox[3]),bbox_color1,2) 

    # #  if not swap_flag:
    # #      cv2.rectangle(im2,(map_bbox[0],map_bbox[1]),(map_bbox[2],map_bbox[3]),map_bbox_color,2)
    # #  if swap_flag:
    # #     cv2.rectangle(im1,(map_bbox[0],map_bbox[1]),(map_bbox[2],map_bbox[3]),map_bbox_color,2)
    # #  cv2.imshow('im1',im1)
    # #  cv2.imshow('im2',im2)

    #  im_path1=os.path.join(res_im_dir,'exp2_im1.jpg')
    #  im_path2=os.path.join(res_im_dir,'exp2_im2.jpg')
    #  cv2.imwrite(im_path1,im1)
    #  cv2.imwrite(im_path2,im2)
    #  cv2.waitKey(-1)
     