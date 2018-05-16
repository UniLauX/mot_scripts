##(1)from cvxopt.modeling import matrix,solvers,variable
# A = matrix([ [-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, -1.0, -2.0] ])
# b = matrix([ 1.0, -2.0, 0.0, 4.0 ])
# c = matrix([ 2.0, 1.0 ])
# sol=solvers.lp(c,A,b)
# print(sol['x'])
   
##(2)Gurobi
##Gurobi
# import sys
# from gurobipy import *   
import numpy as np
from pylpsolve import LP
##http://www.stat.washington.edu/~hoytak/code/pylpsolve/

def demo():
    print 'demo for using lpsolve...'

    print 'using lpsolve to solve the problem..'
    lp=LP()
    ##Specify constaints
    ##(1)x+y<=3
    ##(2)y+z<=4
    lp.addConstraint([[1,1,0],[0,1,2]],"<=",[3,4])
    # Force the first variable to be integer-valued
    ##lp.setInteger(0)

    ##all variables
    lp.setBinary([0,1,2])

    #set objective
    lp.setObjective([1,1,1],mode='minimize')
    
    ##return = lpsolve('set_binary', lp, column, must_be_bin)
    lp.solve()
   
    ##print out the solution
    print lp.getSolution()


def cal_masks_iou(mask1, mask2):
    inter_mask=np.logical_and(mask1, mask2)
    union_mask=np.logical_or(mask1,mask2)    
    mask_iou=(np.sum(inter_mask)+0.0)/np.sum(union_mask)
    return mask_iou


## mask size is the whole image, need to change later.
def shift_mask_iou(mask1,shift_vec,mask2,im_shape,bbox):
    im_width=im_shape[1]
    im_height=im_shape[0]
    s_mask1=np.zeros((im_height,im_width))
    i_mask2=np.zeros((im_height,im_width))
    ##need to consider the case when shift mask(patch) is out of image bounday 
    x1,y1,x2,y2=bbox
    s_x1=max(0, int(bbox[0]+shift_vec[0]))
    s_y1=max(0, int(bbox[1]+shift_vec[1]))
    s_x2=min(im_width-1, int(bbox[2]+shift_vec[0]))
    s_y2=min(im_height-1,int(bbox[3]+shift_vec[1]))
    s_mask1[s_y1:s_y2+1,s_x1:s_x2+1]=mask1
    i_mask2[y1:y2+1,x1:x2+1]=mask2
    s_iou=cal_masks_iou(s_mask1,i_mask2)
    
    return s_iou
   
## get mask_iou base on the whole image(considering the motion)   
def image_mask_iou(mask1,mask2,bbox1,bbox2,shift_vec,im_shape):
    s_iou=0.0
    im_width=im_shape[1]
    im_height=im_shape[0]
    
    s_mask1=np.zeros((im_height,im_width))
    i_mask2=np.zeros((im_height,im_width))
    
    im1_x1,im1_y1,im1_x2,im1_y2=bbox1 
    im2_x1,im2_y1,im2_x2,im2_y2=bbox2

    ##need to consider the case when shift mask(patch) is out of image bounday 
    # s_x1=max(0, int(bbox1[0]+shift_vec[0]))
    # s_y1=max(0, int(bbox1[1]+shift_vec[1]))
    # s_x2=min(im_width-1, int(bbox1[2]+shift_vec[0]))
    # s_y2=min(im_height-1,int(bbox1[3]+shift_vec[1]))
    x1_o=0
    y1_o=0
    x2_o=0
    y2_o=0
    s_x1=int(bbox1[0]+shift_vec[0])
    if s_x1<0:
        x1_o =0-s_x1
        s_x1=0
    s_y1=int(bbox1[1]+shift_vec[1])   
    if s_y1<0:
        y1_o=0-s_y1
        s_y1=0

    s_x2=int(bbox1[2]+shift_vec[0])
    if s_x2> im_width-1:
        x2_o=s_x2-(im_width-1)
        s_x2=im_width-1
    
    s_y2=int(bbox1[3]+shift_vec[1])
    if s_y2> im_height-1:
        y2_o=s_y2-(im_height-1)
        s_y2=im_height-1

    # print s_y2-s_y1+1, s_x2-s_x1+1
    # print (bbox1[3]-y2_o+1)-(bbox1[1]+y1_o), (bbox1[2]-x2_o+1)-(bbox1[0]+x1_o)
    # n_x1=bbox1[0]+x1_o
    # n_y1=bbox1[1]+y1_o
    # n_x2=bbox1[2]-x2_o
    # n_y2=bbox1[3]-y2_o
    s_mask1[s_y1:s_y2+1,s_x1:s_x2+1]=mask1[y1_o:(bbox1[3]-bbox1[1]-y2_o+1),x1_o:(bbox1[2]-bbox1[0]-x2_o+1)]
    i_mask2[im2_y1:im2_y2+1,im2_x1:im2_x2+1]=mask2
    s_iou=cal_masks_iou(s_mask1,i_mask2)
    return s_iou


def get_pairwise_arr(ims,prev_boxes,cur_boxes,next_boxes,fw_masks,cur_masks,bw_masks,shift_arr):
    im_shape=ims[0].shape
    frm_num=len(fw_masks)
    pairwise_arr=[] 
    pairwise_list=[]

    fw_couples=zip(fw_masks,prev_boxes)
    cur_couples=zip(cur_masks,cur_boxes)
    bw_couples=zip(bw_masks,next_boxes)

    for frm_id in xrange(frm_num-1):
        shift_vec=shift_arr[frm_id]

        for maskbox1 in [fw_couples[frm_id],cur_couples[frm_id],bw_couples[frm_id]]:
            mask1,bbox1=maskbox1
            for maskbox2 in [fw_couples[frm_id+1],cur_couples[frm_id+1],bw_couples[frm_id+1]]:
                mask2,bbox2=maskbox2 
                img_mask_iou=image_mask_iou(mask1,mask2,bbox1,bbox2,shift_vec,im_shape)
                pairwise_list.append(img_mask_iou)
    pairwise_arr=np.asarray(pairwise_list)            
    return pairwise_arr

def lpsolve_infer(ims,prev_boxes,cur_boxes,next_boxes,fw_masks,cur_masks,bw_masks,shift_arr):
    picked_boxes=[]
    picked_masks=[]
    y_labels=[]
    ave_iou=[] 
    frm_num=len(fw_masks)

    ## calculate the pairwise items based on maskIoU
    pairwise_arr=get_pairwise_arr(ims,prev_boxes,cur_boxes,next_boxes,fw_masks,cur_masks,bw_masks,shift_arr)  
    
    tmp_masks=[]
    for frm_id in xrange(frm_num):
        tmp_masks.append(fw_masks[frm_id])
        tmp_masks.append(cur_masks[frm_id])
        tmp_masks.append(bw_masks[frm_id])
    all_masks=np.asarray(tmp_masks)

    tmp_bboxes=[]
    for frm_id in xrange(frm_num):
        tmp_bboxes.append(prev_boxes[frm_id])
        tmp_bboxes.append(cur_boxes[frm_id])
        tmp_bboxes.append(next_boxes[frm_id])
    all_boxes=np.asarray(tmp_bboxes)

    ##====================================================== Construct Graph begining============================================================## 
    ##(0)set up parameters:
    edge_num=frm_num*9-3
    node_num=frm_num*3+2
    var_len=edge_num+node_num
    
    ##(1)set node and edge index
    ## 1.1 val_indexes
    val_indexes=np.zeros(var_len,dtype=int)
    for val_id in xrange(var_len):
        val_indexes[val_id]=val_id

    ##------------------------------------------------------------node indexes------------------------------------------------------------ 
    ##1.2 node_indexes     
    node_indexes=np.zeros(node_num,dtype=int)
    node_indexes[0]=0 #source node
    node_indexes[-1]=var_len-1 #sink node

    ##1.2.1 (indexes of(node) in val_indexes)
    for frm_id in xrange(frm_num):
         node_id1=frm_id*12+4

         if frm_id==frm_num-1: ##last frame
             node_id2=node_id1+1
             node_id3=node_id1+2
         else:
             node_id2=node_id1+4 ## other frames
             node_id3=node_id1+8
         t_nodes=[node_id1,node_id2,node_id3]

         ##1.2.2 (indexes of (node)in node_indexes)
         t_id1=(frm_id+1)*3-2
         t_id3=t_id1+2
         node_indexes[t_id1:t_id3+1]=t_nodes 
    
    ##--------------------------------------------------------------edge indexes--------------------------------------------------------
    ##1.3 edge_indexes
    edge_indexes=np.asarray(list(set(val_indexes)-set(node_indexes)))

    ##-------------------------------------------------------------Coeficient vector-----------------------------------------------------
    ##(2) set cofficent vector    
    coef_vec=np.zeros((var_len))

    ##2.1 node cofficient
    coef_vec[node_indexes]=1  ##set all the node(unary term) as 1
    
    ##2.2 edge cofficient
    coef_vec[1:4]=1  #the first three edges linked to the source node
    coef_vec[-4:-1]=1 #the last three edges linked to the sink node

    ##set up pairwise term(edges)
    pairwise_indexes=edge_indexes[3:-3]
    coef_vec[pairwise_indexes]=pairwise_arr
    
    print 'pairwise_indexes:',pairwise_indexes
    print pairwise_indexes.shape
    print 'pairwise_arr:',pairwise_arr
    print pairwise_arr.shape
    

    ##=================================================================Equation constraints========================##
    ## AX=B
    equ_num=2+(frm_num)*3*2   ## 2 for first and last node(with 3 edges) +  frm_num*6 for every frame(3 nodes with each one with 2)

    ##(3) set A Matrix
    a_mat=np.zeros((equ_num,var_len))
     
    ##3.1 
    a_mat[0,1:4]=1  ## three edges flow out from the source node   x1+x2+x3=1
    a_mat[-1,-4:-1]=1 ## three edges flow into the sink node       x(-1)+x(-2)+x(-3)=1

    ##3.2
    ##the first three edges and the three nodes(in the frist frame)
    a_mat[1,1]=a_mat[2,2]=a_mat[3,3]=-1                           #x1=x4, x2=x8, x3=x12
    a_mat[1,4]=a_mat[2,8]=a_mat[3,12]=1                           

    ##the last three edges and the three nodes(in the last frame)
    a_mat[equ_num-4,var_len-4]=a_mat[equ_num-3,var_len-3]=a_mat[equ_num-2,var_len-2]=1    #x(-1)=x(-4), x(-2)=x(-8), x(-3)=x(-12)
    a_mat[equ_num-4,var_len-7]=a_mat[equ_num-3,var_len-6]=a_mat[equ_num-2,var_len-5]=-1
 
    ##3.3
    ##outflow constraints(1 node to 3 edges)
    row_idx=4
    for frm_id in xrange(frm_num-1):
        t_base=frm_id*3+1
        for idx in xrange(3):
            t_id=[t_base+idx]
            src_node_idx=node_indexes[t_id]
            brh_edge_idxes=val_indexes[int(src_node_idx+1):int(src_node_idx+4)]
            a_mat[row_idx,src_node_idx]=-1
            a_mat[row_idx,brh_edge_idxes]=1
            row_idx=row_idx+1

    ##3.3
    ##inflow constraints(3 edges to 1 node)
    for frm_id in xrange(frm_num-1):
        t_edge_base=frm_id*9+3

        frm_id1=frm_id+1
        t_node_base=frm_id1*3+1  ##node base
        for idx in xrange(3):
            t_node_id=[t_node_base+idx]
            dest_node_idx=node_indexes[t_node_id]
            t_edge_id=[t_edge_base+idx]
            src_edge_id=edge_indexes[t_edge_id][0]
            src_edge_idxes=np.asarray([src_edge_id,src_edge_id+4,src_edge_id+8])
            a_mat[row_idx,dest_node_idx]=-1
            a_mat[row_idx,src_edge_idxes]=1 
            row_idx=row_idx+1
  
    ##4. set B Vector
    b_vec=np.zeros(equ_num)
    b_vec[0]=b_vec[-1]=1    

    ##===============================================================================Solvers=========================================================
    lp=LP()
    lp.addConstraint(a_mat,"=",b_vec)

    # ##all variables
    lp.setBinary(val_indexes)

    # #set objective
    lp.setObjective(coef_vec,mode='maximize')
    
    lp.solve()
    val_y=lp.getSolution()


    ##----------------------------------------------------------------------------------------------------------------------------------------
    ## map choosen nodes
    chosen_nodes=val_y[node_indexes]
    y=np.where(chosen_nodes[1:-1]>0)[0]
    y_labels=np.asarray(y)
    picked_masks=all_masks[y_labels]
    picked_boxes=all_boxes[y_labels]    

    ## calculate the edge iou.
    chosen_edges=val_y[edge_indexes]
    e=np.where(chosen_edges[3:-3]>0)[0]
    e_labels=np.asarray(e)
    ave_iou=np.sum(pairwise_arr[e_labels])/(frm_num-1)
    print 'average iou:',ave_iou
    print 'y_label:', y_labels
    return picked_boxes,picked_masks, y_labels,ave_iou   


if __name__ == '__main__':
    lpsolve_infer()
   