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

# Add toolkit to PYTHONPATH
common_path = osp.join(this_dir, '..', 'toolkit')
add_path(common_path)
