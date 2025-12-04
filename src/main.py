import os
import wfdb
import random
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.interpolate import interp1d

from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve