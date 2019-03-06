# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 20:36:33 2018

@author: Dylan
"""

from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
def guess_image_dim(in_shape):
    side_len = int(np.sqrt(in_shape))
    if np.abs(in_shape-side_len*side_len)<2:
        return (int(side_len), int(side_len))
    else:
        side_len = int(np.sqrt(in_shape/3))
        return (side_len, side_len, 3)
csv_dir = os.path.join('..', 'input', 'dermatology-mnist-loading-and-processing')