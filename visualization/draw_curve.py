## This one is mainly for drawing curve of average maskIoU,precision-reall curves.
#  target dataset is tud_crossing(201 images in total)
#  target instances are 5, 10, 12, 13, 14 (since they are continious in gt_annots)  

import numpy as np

from pylab import *
import matplotlib.patches as mpatches

## average maskIoU curve
def ave_maskIoU_curve():
    corrup_ratios=np.array([0.1,0.3,0.5,0.7,0.9])
    corrup_maskiou=np.array([0.9597561936,0.8889043409,0.8118235544,0.7272874819,0.6721927124])
    refine_maskiou=np.array([0.8856055047,0.8750119054,0.8349197446,0.7454941775,0.7332714232])

    _, ax1 = subplots()
    ax1.plot(corrup_ratios,corrup_maskiou,'r')
    ax1.plot(corrup_ratios,refine_maskiou,'b')

    red_patch = mpatches.Patch(color='red', label='corrup mask') 
    blue_patch = mpatches.Patch(color='blue', label='refine mask')
    
    plt.legend(handles=[red_patch,blue_patch])
       #method1: draw on line 
    ax1.set_xlabel('corrup_ratio')
    ax1.set_ylabel('average maskIoU')
    ax1.set_title('average maskIoU curve') 
    plt.show()


def ave_recall_curve():
    corrup_ratios=np.array([0.1,0.3,0.5,0.7,0.9])

    corrup_recall=np.array([0.9697986577,0.9093959732,0.8456375839,0.7785234899,0.7382550336])

    refine_recall=np.array([0.9630872483,0.9463087248,0.9228187919,0.8758389262,0.8791946309])
   
    _, ax1 = subplots()
    ax1.plot(corrup_ratios,corrup_recall,'r')
    ax1.plot(corrup_ratios,refine_recall,'b')

    red_patch = mpatches.Patch(color='red', label='corrup mask') 
    blue_patch = mpatches.Patch(color='blue', label='refine mask')
    
    plt.legend(handles=[red_patch,blue_patch])
       #method1: draw on line 
    ax1.set_xlabel('corrup_ratio')
    ax1.set_ylabel('average recall')
    ax1.set_title('average recall curve(maskIoU_thre=0.5)') 
    plt.show()



## main func
if __name__ == '__main__':
    print 'draw curve...'
    ##ave_maskIoU_curve()
    ave_recall_curve()
   