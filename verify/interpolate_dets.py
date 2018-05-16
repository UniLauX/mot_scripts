import _init_paths
from tud_crossing import load_dets_new


from common.generate_util import interploate_dets
import os

debug_flag=True

gt_annots_dir='/mnt/phoenix_fastdir/dataset/TUD-Crossing/gt_new'

## main func
if __name__ == '__main__':
    print '=================interpolate dets===================='
    gt_dets=load_dets_new()
    disc_dets_path=os.path.join(gt_annots_dir,'dets.txt')
    cont_dets_path=os.path.join(gt_annots_dir,'cont_dets.txt')
    interploate_dets(disc_dets_path,cont_dets_path)

    if debug_flag:
        print 'gt_dets:', gt_dets.shape

