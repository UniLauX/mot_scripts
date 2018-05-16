import numpy as np
import matplotlib.pyplot as plt
from time import time
from pystruct.inference import inference_dispatch, compute_energy
from pystruct.utils import make_grid_edges

def test_demo():
     frm_num=3
     size=frm_num
     n_states=2
     unaries= np.array([[0.6, 0.5], [1, 0], [.4, .6]])
     edges= np.array([[0, 1], [1, 2]])
     pairwise=np.array([[0, 0], [0, 0]])

     print unaries.shape
     print edges.shape
     print pairwise.shape
      
     fig, ax = plt.subplots(1, 2, figsize=(3, 1))

     ##for a, inference_method in zip(ax, ['ad3', 'qpbo', 'max-product',
     ##                               ('max-product', {'max_iter': 10}), 'lp']):
     for a, inference_method in zip(ax, ['lp']):                               
                                    start = time()
                                    print a, inference_method
                                    y = inference_dispatch(unaries, pairwise, edges,
                                    inference_method=inference_method)    ##(400)
                                    took = time() - start

                                    a.matshow(y.reshape(size, 1))
                                    print y.shape
                                    print y
                                    energy = compute_energy(unaries, pairwise, edges, y)
                                    a.set_title(str(inference_method) + "\n time: %.2f energy %.2f" % (took, energy))
                                    ##a.set_xticks(())
                                    ##a.set_yticks(())
                                    plt.show()

def crf_infer(fw_masks,cur_masks,bw_masks):
     print '================================using CRF to do inference=========================================================='
     frm_num=len(cur_masks)
     n_nodes=frm_num
     states_per_frm=3
     n_states=frm_num*states_per_frm
    
     mask_sample=cur_masks[0]
     w_mask=mask_sample.shape[1]
     h_mask=mask_sample.shape[0]
     area_mask=w_mask*h_mask    ## area_mask is fixed here, but can be different later. 
     
     mask1=fw_masks[0]
     mask2=cur_masks[0]

     match_mask=np.logical_and(mask1, mask2)
     
     ##np.logical_and([True, False], [False, False])

     ##put all three kinds of mask in one category
     state_masks=[]
     for node_id in xrange(n_nodes):
         state_masks.append(fw_masks[node_id])
         state_masks.append(cur_masks[node_id]) 
         state_masks.append(bw_masks[node_id])
     
     ##(1) define edges 
     edges=np.zeros((n_nodes-1,2),dtype=int) 
     ##edges=np.zeros((2,2),dtype=int)    ##(node-1,2) 
     for node_id in xrange(n_nodes-1):
         edges[node_id,:]=[node_id,node_id+1]   

     ##(2) define unaries
     ## calculate unary values within the same frame
    #  unary_vals=np.zeros((n_nodes,n_states))
    #  for node_id in xrange(n_nodes):
    #      mask1=fw_masks[node_id]
    #      mask2=cur_masks[node_id]
    #      mask3=bw_masks[node_id]
    #      inter_1_2=inter_2_1=np.logical_and(mask1,mask2)
    #      inter_2_3=inter_3_2=np.logical_and(mask2,mask3)
    #      inter_1_3=inter_3_1=np.logical_and(mask1,mask3)

    #      val1=np.sum(inter_1_2)+np.sum(inter_1_3)
    #      val2=np.sum(inter_2_1)+np.sum(inter_2_3)
    #      val3=np.sum(inter_3_1)+np.sum(inter_3_2)
         
    #      val_sum=val1+val2+val3+0.0
    #      norm_val1=val1/val_sum
    #      norm_val2=val2/val_sum
    #      norm_val3=val3/val_sum
         
    #      t_nodes=[node_id*states_per_frm,node_id*states_per_frm+1,node_id*states_per_frm+2]
    #      unary_vals[node_id,t_nodes]=[norm_val1,norm_val2,norm_val3]
    #  unaries=unary_vals


    ##ignore unary terms:
     unary_vals=np.ones((n_nodes,n_states))
     unaries=unary_vals

      #  ##(2) define unaries
     #unaries=np.zeros((n_nodes,n_states))
     #  for node_id in xrange(n_nodes):
     #      t_nodes=[node_id*states_per_frm,node_id*states_per_frm+1,node_id*states_per_frm+2]
     #      unaries[node_id,t_nodes]=1.0/3
     
     ##(3) define pairwise
     pairwise=np.zeros((n_states, n_states))
     
     t_base=np.zeros((n_nodes,n_nodes))

     for state_id1 in xrange(n_states):
         t1=state_id1/states_per_frm
         for state_id2 in xrange(n_states):
             t2=state_id2/states_per_frm
             if t1==t2 or abs(t2-t1)>1 :
                 continue
             else:
                 mask1=state_masks[state_id1]
                 mask2=state_masks[state_id2]
                 iou_1_2=np.logical_and(mask1,mask2)
                 val_1_2=np.sum(iou_1_2)
                 pairwise[state_id1,state_id2]=val_1_2
                 t_base[t1,t2]=val_1_2+t_base[t1,t2]

      ##s1: normalize  based on two consective frames(9 pairs)
     for state_id1 in xrange(n_states):
        t1=state_id1/states_per_frm
        for state_id2 in xrange(n_states):
            t2=state_id2/states_per_frm
            if t1==t2 or abs(t2-t1)>1 :
                continue
            else:
                pairwise[state_id1,state_id2]=pairwise[state_id1,state_id2]/t_base[t1,t2]
     
     pairwise=pairwise
   
       ##s2: normalize on the whole sequence.

     ##print 't_base:\n', t_base
     ##print 'unaries:\n',unaries
     ##print 'pairwise:\n',pairwise

     print 'unaries.shape:',unaries.shape
     print 'edges.shape:',edges.shape
     print 'pairwise.shape:',pairwise.shape
      
     fig, ax = plt.subplots(1, 2, figsize=(n_nodes, 1))

     ##for a, inference_method in zip(ax, ['ad3', 'qpbo', 'max-product',
     ##                               ('max-product', {'max_iter': 10}), 'lp']):
     for a, inference_method in zip(ax, ['lp']):                               
                                    start = time()
                                    ##print a, inference_method
                                    y = inference_dispatch(unaries, pairwise, edges,
                                    inference_method=inference_method)    ##(400)
                                    took = time() - start
                                    a.matshow(y.reshape(n_nodes, 1))
                                    ##print y.shape
                                    energy = compute_energy(unaries, pairwise, edges, y)
                                    ##a.set_title(str(inference_method) + "\n time: %.2f energy %.2f" % (took, energy))
                                    ##a.set_xticks(())
                                    ##a.set_yticks(())
                                    ##plt.show()

     arr_state_masks=np.asarray(state_masks)
     picked_masks= arr_state_masks[y]                       
     return y,picked_masks                               

if __name__ == '__main__':
     print "test GraphCRF.."
     ##test_demo()
  
   


                                   