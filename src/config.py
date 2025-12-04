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


@dataclass
class TrainCfg:
    window_sec = 2.0
    crop_len = 360
    records = tuple([
        '100','101','102','103','104','105','106','107','108','109','111','112','113','114','115',
        '116','117','118','119','121','122','123','124','200','201','202','203','205','207','208',
        '209','210','212','213','214','215','217','219','220','221','222','223','228','230','231',
        '232','233','234'
    ])
    label_keep = ("N", "V", "F")
    split_mode = "stratified"
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    crop_len: int = 720
    batch_size: int = 128
    max_epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    use_sampler: bool = True
    sampler_scale: float = 0.2
    focal_gamma: float = 1.5
    hybrid_switch_epoch: int = 15
    augment: bool = True
    severity: int = 0.25
    mixup_p: float = 0.5
    mixup_alpha: float = 0.1
    clip_grad_norm: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    save_dir: str = "checkpoints"
    seed: int = 42

CFG = TrainCfg()

torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)
random.seed(CFG.seed)