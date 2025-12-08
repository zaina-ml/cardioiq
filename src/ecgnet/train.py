from config import CFG
from data import ECGDataset
from utils import tune_threshold_from_probs


import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve

class FocalLoss(nn.Module):
    def __init__(self, focal_gamma=None, pos_weight=None):
        super().__init__()

        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets, epoch=0):
        prob = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t = prob * targets + (1 - prob) * (1 - targets)
        focal = ((1 - p_t) ** self.focal_gamma) * bce
        if self.pos_weight is not None:
            w = torch.where(targets == 1, self.pos_weight, torch.tensor(1.0, device=targets.device))
            focal = focal * w
        return focal.mean()

class BCEWeighted(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()

        self.pos_weight = pos_weight

    def forward(self, logits, targets, epoch=0):
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="mean"
        )



def train_model(model: nn.Module,
                X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                cfg):
    device = cfg.device
    model.to(device)

    train_ds = ECGDataset(X_train, y_train, augment=True, cfg=cfg)
    val_ds   = ECGDataset(X_val,   y_val, augment=False, cfg=cfg)

    labels_flat = y_train.astype(int).flatten()
    class_counts = np.bincount(labels_flat)
    print("Class counts:", class_counts)

    if cfg.use_sampler:
        weights = 1. / class_counts
        sample_weights = weights[y_train.astype(int)]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    pos_weight = torch.tensor(
        (len(y_train) - y_train.sum()) / max(1, y_train.sum()),
        dtype=torch.float32,
        device=device
    )

    criterion = FocalLoss(focal_gamma=cfg.focal_gamma, pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=cfg.lr * 1e-3
    )

    scaler = amp.GradScaler("cuda", enabled=cfg.amp and device.startswith("cuda"))

    best_f1, best_state = -1.0, None
    smoothed_thresh = 0.5
    alpha_thresh = 0.3

    for epoch in range(1, cfg.max_epochs + 1):
        train_ds.current_epoch = epoch
        model.train()
        total_loss, n_seen = 0.0, 0
        severity_sum = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).view(-1)
            optimizer.zero_grad()

            batch_severity = np.mean([train_ds.scheduler.get(epoch, int(y.item())) for y in yb])
            severity_sum += batch_severity * xb.size(0)

            with amp.autocast('cuda', enabled=scaler.is_enabled()):
                logits = model(xb).view(-1)
                loss = criterion(logits, yb, epoch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)

        scheduler.step(epoch + 1)

        train_loss = total_loss / max(1, n_seen)
        avg_severity = severity_sum / max(1, n_seen)

        model.eval()
        all_probs, all_y = [], []
        with torch.inference_mode(), amp.autocast('cuda', enabled=scaler.is_enabled()):
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb).view(-1)
                probs = torch.clamp(torch.sigmoid(logits), min=1e-7, max=1 - 1e-7).cpu().numpy()
                all_probs.append(probs)
                all_y.append(yb.numpy().astype(int).flatten())

        all_probs = np.concatenate(all_probs)
        all_y = np.concatenate(all_y)

        pr_auc = float(average_precision_score(all_y, all_probs))
        roc_auc = float(roc_auc_score(all_y, all_probs))
        epoch_thresh, epoch_f1 = tune_threshold_from_probs(all_y, all_probs)

        preds = (all_probs >= epoch_thresh).astype(int)
        epoch_precision = precision_score(all_y, preds, zero_division=0)
        epoch_recall = recall_score(all_y, preds, zero_division=0)

        smoothed_thresh = alpha_thresh * epoch_thresh + (1 - alpha_thresh) * smoothed_thresh

        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            ckpt_path = os.path.join(cfg.save_dir, f"best_f1_epoch{epoch:03d}_f1{best_f1:.4f}.pt")
            torch.save({
                'model_state': best_state,
                'epoch': epoch,
                'f1': best_f1,
                'threshold': smoothed_thresh
            }, ckpt_path)

        print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} "
              f"| best_F1 {best_f1:.4f} @thr {smoothed_thresh:.3f} "
              f"(Precision {epoch_precision:.3f} | Recall {epoch_recall:.3f} | PR-AUC {pr_auc:.3f} | ROC-AUC {roc_auc:.3f} | AvgSeverity {avg_severity:.3f})")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Training done. Best F1: {best_f1:.4f} | Smoothed threshold: {smoothed_thresh:.3f}")
    return model, smoothed_thresh, best_f1