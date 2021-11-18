# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.core.config import cfg
from pysot.tracker.siamrpn_tracker import SiamRPNTracker

from pysot.tracker.siammask_tracker import SiamMaskTracker
from pysot.tracker.siamrpnlt_tracker import SiamRPNLTTracker
from pysot.tracker.dsiamrpn_tracker import DSiamRPNTracker
from pysot.tracker.dimp_tracker import DiMPTracker
from pysot.tracker.kys_tracker import KYSTracker
from pysot.tracker.cf_tracker import CFTracker
from pysot.tracker.eco_tracker import ECOTracker


#from run_tracker import  p_config
import os
import pickle
import importlib
TRACKS = {
          'SiamRPNTracker': SiamRPNTracker,
          'SiamMaskTracker': SiamMaskTracker,
          'SiamRPNLTTracker': SiamRPNLTTracker,
          'DSiamRPNTracker': DSiamRPNTracker,
          'DiMPTracker': DiMPTracker,
          'CFTracker':CFTracker,
          'KYSTracker': KYSTracker,
          'ECOTracker':ECOTracker,
        
            #'DiMP_LT': Dimp_LTMU_Tracker
         }


def build_tracker(model,pth=None):
    if cfg.TRACK.TYPE in ['DiMPTracker','ATOMTracker','ECOTracker','KYSTracker']:
        params = get_parameters()
        #print(params)
        #input()
        return TRACKS[cfg.TRACK.TYPE](params)

    if cfg.TRACK.TYPE=='CFTracker':
        return TRACKS[cfg.TRACK.TYPE]()
   
    else:
        return TRACKS[cfg.TRACK.TYPE](model)

def get_parameters():
    """Get parameters."""

    parameter_file = '{}/parameters.pkl'.format(cfg.PYTRACKING.PARAM_DIR)
    if os.path.isfile(parameter_file):
        return pickle.load(open(parameter_file, 'rb'))
    print(cfg.PYTRACKING.PARAM_NAME)
    input()
    param_module = importlib.import_module('extern.pytracking.parameter.{}.{}'.format(cfg.PYTRACKING.TRACKER_NAME, cfg.PYTRACKING.PARAM_NAME))
    params = param_module.parameters()

    #pickle.dump(params, open(parameter_file, 'wb'))

    return params