import numpy as np

import scipy.io as sio


def load_data( data_path):
     ##
    data_arr=[]
    with open(src_points_path) as f:
        data_rows= [x.strip() for x in f.readlines()]
        for rowtxt in data_rows: 
            data_arr.append(rowtxt.split(','))
    #gtarr = np.array(gtarr, dtype=np.float32)    #transfer list to np.array
    return data_arr

def visual_tracking_features():
    print 'visualized tracking features...'

if __name__ == '__main__':
    print 'load point tracking...'
    ##src_points_path='/home/uni/Lab/projects/C++/trackingCPU/cars1Results/trackRes/4PointSampling/cars1Tracks10for10frms.dat'
    src_points_path='/home/uni/Lab/projects/C++/trackingCPU/cars1Results/trackRes/8PointSampling/cars1Tracks10for10frms.dat'
    dest_points_path='/home/uni/Lab/projects/C++/trackingCPU/cars1Results/trackRes/transfered_10frm_8pt.mat'
    
    data_arr=load_data(src_points_path)
   
    frame_number=int(data_arr[0][0])
    point_number=int(data_arr[1][0])
    
    points_mat=np.zeros([frame_number*2,point_number])  # tracking points:(shape=2F*D)
    label_mat=np.zeros([point_number,1])*-1
    print label_mat.shape

    
    track_arr=data_arr[2:]  

    track_row_idx=0
    track_col_idx=-1

    for rec_row_idx in xrange(len(track_arr)):
        if rec_row_idx % (frame_number+1)==0:
            
            track_col_idx=track_col_idx+1
            track_row_idx=0
            ##read the n(th) point information
            print '-------------------------------'
            for sub_idx in xrange(frame_number):
                track_row= track_arr[rec_row_idx+sub_idx+1]
                track_row=np.array(track_row)[0]
                track_row=track_row.split(' ')
                x=float(track_row[0])
                y=float(track_row[1])
                print 'row:',track_row_idx, ' col:', track_col_idx
                print 'x:',x, 'y:',y
                points_mat[track_row_idx][track_col_idx]=x
                points_mat[track_row_idx+1][track_col_idx]=y
                track_row_idx=track_row_idx+2

    sio.savemat(dest_points_path,{'X':points_mat,'s':label_mat})

    print points_mat[:,:6]
    
