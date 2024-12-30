''' For tutorial notebok.'''

# Get system path
import os
current_dir = os.path.dirname(os.path.abspath('dependencies.py'))
current_dir = os.path.join(current_dir, 'utils')

# System tools
import sys
sys.path.append(current_dir)

# Self tools
from model_utils import *
from algorithm_utils import *
from output_utils import *

# Sim tools
from transform_utils import projection, reconstruction

# Model tools
import torch
import torch.nn as nn

# Utils
import numpy as np
import matplotlib.pyplot as pyp
import matplotlib.pylab as plt

# Console capture
import contextlib