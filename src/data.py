from config import CFG

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

def load_record(name, duration_sec=600):
    rec = wfdb.rdrecord(name, pn_dir="mitdb")
    ann = wfdb.rdann(name, "atr", pn_dir="mitdb")
    fs = rec.fs
    sig = rec.p_signal[:int(fs*duration_sec)]
    valid = ann.sample < int(fs*duration_sec)
    ann.sample = ann.sample[valid]
    ann.symbol = np.array(ann.symbol)[valid].tolist()
    return sig, ann, fs

def segment_beats(signal, ann, fs, window_sec, label_keep):
    half = int(fs*window_sec/2)
    beats, labels = [], []
    for sample, sym in zip(ann.sample, ann.symbol):
        if sym not in label_keep:
            continue
        start, end = sample-half, sample+half
        if start<0 or end>len(signal):
            continue
        seg = signal[start:end,0]
        seg = (seg - np.mean(seg))/(np.std(seg)+1e-8)
        beats.append(seg.astype(np.float32))
        labels.append(0 if sym=="N" else 1)
    return np.array(beats), np.array(labels)

from sklearn.model_selection import train_test_split

def build_dataset(cfg):
    if getattr(cfg, "split_mode", "record") == "record":
        rng = np.random.default_rng(cfg.seed)
        records = list(cfg.records)
        rng.shuffle(records)
        n_total = len(records)
        n_train = int(n_total * cfg.train_ratio)
        n_val = int(n_total * cfg.val_ratio)
        train_recs = records[:n_train]
        val_recs = records[n_train:n_train + n_val]
        test_recs = records[n_train + n_val:]

        def proc(rec_list):
            xs, ys = [], []
            for r in rec_list:
                sig, ann, fs = load_record(r)
                x, y = segment_beats(sig, ann, fs, cfg.window_sec, cfg.label_keep)
                if x.size == 0 or y.size == 0:
                    continue
                if x.ndim == 1:
                    x = x[None, :]
                xs.append(x[:, None, :])
                ys.append(y)
            if len(xs) == 0:
                return np.empty((0, 1, cfg.crop_len)), np.empty((0,), dtype=int)
            return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

        X_train, y_train = proc(train_recs)
        X_val, y_val = proc(val_recs)
        X_test, y_test = proc(test_recs)

    else:
        xs, ys = [], []
        for r in cfg.records:
            sig, ann, fs = load_record(r)
            x, y = segment_beats(sig, ann, fs, cfg.window_sec, cfg.label_keep)
            if x.size == 0 or y.size == 0:
                continue
            if x.ndim == 1:
                x = x[None, :]
            xs.append(x[:, None, :])
            ys.append(y)
        X = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=cfg.val_ratio + cfg.test_ratio,
            stratify=y,
            random_state=cfg.seed
        )
        rel_test = cfg.test_ratio / (cfg.val_ratio + cfg.test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=rel_test,
            stratify=y_temp,
            random_state=cfg.seed
        )
    print("Class distribution:")
    for name, labels in zip(["Train", "Val", "Test"], [y_train, y_val, y_test]):
        if len(labels) > 0:
            ratio = np.mean(labels)
            print(f"{name}: {len(labels)} samples | pos ratio {ratio:.4f}")
        else:
            print(f"{name}: 0 samples")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class AugmentationScheduler:
    def __init__(self, prog_epochs=10, max_severity=None):
        self.prog_epochs = prog_epochs
        self.max_severity = max_severity

    def get(self, epoch, y):
        base = min(1.0, epoch / max(1, self.prog_epochs))
        severity = base * self.max_severity

        return min(severity, self.max_severity)


class AddGaussianNoise(nn.Module):
    def __init__(self, std_factor=0.05):
        super().__init__()
        self.std_factor = std_factor

    def forward(self, x, severity):
        std = torch.std(x) * self.std_factor * severity * 10
        return x + torch.randn_like(x) * std


class AmplitudeScale(nn.Module):
    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale

    def forward(self, x, severity):
        factor = 1 + self.scale * (2 * torch.rand(1, device=x.device) - 1) * severity
        return x * factor


class TimeWarp(nn.Module):
    def __init__(self, max_stretch=0.05):
        super().__init__()
        self.max_stretch = max_stretch

    def forward(self, x, severity):
        L = x.shape[-1]

        stretch = 1 + (torch.rand(1, device=x.device) * 2 - 1) * self.max_stretch * severity
        new_len = max(2, int(L * stretch.item()))

        x_in = x.unsqueeze(0).unsqueeze(0)

        warped = F.interpolate(x_in, size=new_len, mode="linear", align_corners=False)
        resampled = F.interpolate(warped, size=L, mode="linear", align_corners=False)

        return resampled.squeeze(0).squeeze(0)


class RandomShift(nn.Module):
    def __init__(self, max_shift=0.05):
        super().__init__()
        self.max_shift = max_shift

    def forward(self, x, severity):
        L = x.shape[-1]
        max_shift = int(L * self.max_shift * severity)
        if max_shift > 0:
            k = int(torch.randint(-max_shift, max_shift + 1, (1,), device=x.device))
            x = torch.roll(x, shifts=k, dims=-1)
        return x


class RandomDropout(nn.Module):
    def __init__(self, max_frac=0.05):
        super().__init__()
        self.max_frac = max_frac

    def forward(self, x, severity):
        L = x.shape[-1]
        chunk_len = max(1, int(L * self.max_frac * severity))
        if chunk_len > 0:
            start = int(torch.randint(0, max(1, L - chunk_len), (1,), device=x.device))
            x[start:start + chunk_len] = 0.0
        return x


class RandomSpike(nn.Module):
    def __init__(self, max_amp=0.2):
        super().__init__()
        self.max_amp = max_amp

    def forward(self, x, severity):
        L = x.shape[-1]
        amp = self.max_amp * severity * torch.std(x)
        at = int(torch.randint(0, L, (1,), device=x.device))
        l = min(int(torch.randint(1, 4, (1,), device=x.device)), L - at)
        x[at:at + l] += torch.randn(l, device=x.device) * amp
        return x


class ECGAugment(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x, severity):
        chosen = random.sample(list(self.transforms), random.randint(1, len(self.transforms)))
        for t in chosen:
            x = t(x, severity)
        return torch.clamp(x, -5, 5)


class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, augment=False, cfg=None):
        self.X = torch.tensor(X.squeeze(), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        self.cfg = cfg
        self.augment = augment
        self.severity = cfg.severity

        self.scheduler = AugmentationScheduler(
            max_severity=self.severity,
        )
        self.current_epoch = 0

        self.augment_pipeline = ECGAugment([
            AddGaussianNoise(),
            AmplitudeScale(),
            TimeWarp(),
            RandomShift(),
            RandomDropout(),
            RandomSpike()
        ])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.augment:
            severity = self.scheduler.get(self.current_epoch, int(y.item()))
            x = self.augment_pipeline(x.to(self.cfg.device), severity)

        return x.unsqueeze(0), y
