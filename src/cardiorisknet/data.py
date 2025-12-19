import torch
from torch.utils.data import Dataset
from config import CFG
import random

class SyntheticPatientDataset(Dataset):
    def __init__(self, num_samples=2000, cfg=CFG):
        self.features = []
        self.targets = []

        for _ in range(num_samples):
            ecg_prob = random.uniform(0, 1)
            exercise = random.uniform(0, 1)
            diet = random.uniform(0, 1)
            sleep = random.uniform(0, 1)

            smoking_per_week = random.uniform(0, 140)
            alcohol_per_week = random.uniform(0, 30)

            age = random.uniform(20, 80)
            sex = random.choice([0, 1])
            bmi = random.uniform(18.5, 35)

            sys_bp = random.uniform(90, 180)
            dia_bp = random.uniform(60, 120)

            age_norm = (age - 20) / 60
            bmi_norm = (bmi - 18.5) / 16.5
            sys_bp_norm = (sys_bp - 90) / 90
            dia_bp_norm = (dia_bp - 60) / 60
            smoking_norm = smoking_per_week / 140
            alcohol_norm = alcohol_per_week / 30

            feat = [
                ecg_prob,
                exercise,
                diet,
                sleep,
                smoking_norm,
                alcohol_norm,
                age_norm,
                sex,
                bmi_norm,
                sys_bp_norm,
                dia_bp_norm
            ]

            self.features.append(torch.tensor(feat, dtype=torch.float32))

            risk = (
                0.5 * ecg_prob +
                0.1 * (1 - exercise) +
                0.1 * (1 - diet) +
                0.1 * (1 - sleep) +
                0.1 * smoking_norm +
                0.1 * alcohol_norm +
                0.15 * age_norm +
                0.15 * sys_bp_norm +
                0.1 * dia_bp_norm +
                0.1 * bmi_norm
            )

            risk = max(0.0, min(risk, 1.0))
            self.targets.append(torch.tensor(risk, dtype=torch.float32))

        self.features = torch.stack(self.features).to(cfg.device)
        self.targets = torch.stack(self.targets).to(cfg.device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
