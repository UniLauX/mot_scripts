import cv2
import os

def im_seq_to_video(algr_name,im_dir,im_names,im_ext,video_dir,video_fps):
    print 'image sequence to video...'

    print im_dir
    print len(im_names)
    print im_ext
    print video_dir
    print video_fps

    im0_path=os.path.join(im_dir,im_names[0]+im_ext)
    im0=cv2.imread(im0_path)
    print im0_path
    print im0.shape

    im_height=im0.shape[0]
    im_width=im0.shape[1]

     #define Videowriter
    fourcc=cv2.cv.CV_FOURCC('M','J','P','G') 
    
    output_name=os.path.join(video_dir,algr_name+'_'+str(int(video_fps))+'fps.avi')
    out = cv2.VideoWriter(output_name, fourcc, video_fps, (im_width,im_height))      
        ################################################(1)
    for im_name in im_names:
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            img_path=os.path.join(im_dir,im_name+im_ext)
            frame = cv2.imread(img_path)
            #frame=cv2.resize(frame,(im_width,im_height))
            out.write(frame) # Write out frame to video
            print 'frame:',img_path
        #     cv2.imshow('video',frame)
        #     if (cv2.waitKey(1) & 0xFF) == ord('q'):
        #             break                                   
    out.release()
    cv2.destroyAllWindows()




