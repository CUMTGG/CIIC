from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import torch


from .AttModel import *

from .TransformerCIIC import CIIC


def setup(opt):
    if opt.caption_model in ['fc', 'show_tell']:
        print('Warning: %s model is mostly deprecated; many new features are not supported.' %opt.caption_model)
        if opt.caption_model == 'fc':
            print('Use newfc instead of fc')

    elif opt.caption_model == 'CIIC':
         model = CIIC(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    return model
