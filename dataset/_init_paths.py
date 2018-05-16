"""Set up paths for MOT-SCRIPTS."""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add configration to PYTHONPATH
config_path = osp.join(this_dir, '..', 'configration')
add_path(config_path)

# Add common to PYTHONPATH
common_path = osp.join(this_dir, '..', 'common')
add_path(common_path)

# Add IO to PYTHONPATH
io_path = osp.join(this_dir, '..', 'IO')
add_path(io_path)


# Add visualization to PYTHONPATH
visual_path = osp.join(this_dir, '..', 'visualization')
add_path(visual_path)


# Add visualization to PYTHONPATH
mot_imgnet_path = osp.join(this_dir, '..', 'mot_imgnet')
add_path(mot_imgnet_path)


# Add flow_tracking to PYTHONPATH
flow_track_path = osp.join(this_dir, '..', 'flow_tracking')
add_path(flow_track_path)


# Add verify to PYTHONPATH
verify_path = osp.join(this_dir, '..', 'verify')
add_path(verify_path)


