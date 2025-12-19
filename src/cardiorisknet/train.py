import torch
from torch import nn
from torch import optim

from torch.utils.data import DataLoader
from model import CardioRiskNet
from data import SyntheticPatientDataset


class SyntheticPatientDataset(Dataset):
    def __init__(self, num_samples=5000, device="cpu"):
        self.X = []
        self.y = []

        for _ in range(num_samples):
            r = random.random()

            # -------- Regime Sampling --------
            if r < 0.25:  # LOW RISK
                ecg_prob = random.uniform(0.0, 0.1)
                exercise = random.uniform(0.85, 1.0)
                diet = random.uniform(0.85, 1.0)
                sleep = random.uniform(0.85, 1.0)
                smoking = random.randint(0, 5)
                alcohol = random.randint(0, 3)
                age = random.uniform(20, 35)
                sys_bp = random.uniform(100, 120)
                dia_bp = random.uniform(65, 80)
                height_cm = random.uniform(160, 185)
                weight_kg = random.uniform(55, 80)

            elif r < 0.75:  # MID RISK
                ecg_prob = random.uniform(0.2, 0.6)
                exercise = random.uniform(0.4, 0.7)
                diet = random.uniform(0.4, 0.7)
                sleep = random.uniform(0.4, 0.7)
                smoking = random.randint(5, 60)
                alcohol = random.randint(3, 15)
                age = random.uniform(35, 60)
                sys_bp = random.uniform(120, 150)
                dia_bp = random.uniform(80, 95)
                height_cm = random.uniform(155, 185)
                weight_kg = random.uniform(70, 100)

            else:  # HIGH RISK
                ecg_prob = random.uniform(0.7, 1.0)
                exercise = random.uniform(0.0, 0.3)
                diet = random.uniform(0.0, 0.3)
                sleep = random.uniform(0.0, 0.3)
                smoking = random.randint(60, 140)
                alcohol = random.randint(15, 30)
                age = random.uniform(60, 80)
                sys_bp = random.uniform(150, 180)
                dia_bp = random.uniform(95, 120)
                height_cm = random.uniform(155, 185)
                weight_kg = random.uniform(90, 130)

            sex = random.choice([0, 1])  # 0 = male, 1 = female

            age_n = (age - 20) / 60
            smoke_n = min(smoking / 140, 1.0)
            alcohol_n = min(alcohol / 30, 1.0)
            sys_n = (sys_bp - 90) / 90
            dia_n = (dia_bp - 60) / 60

            bmi = weight_kg / (height_cm / 100) ** 2
            bmi_n = max(0.0, min((bmi - 18.5) / 16.5, 1.0))

            base_risk = (
                0.40 * ecg_prob +
                0.10 * age_n +
                0.10 * bmi_n +
                0.10 * sys_n +
                0.05 * dia_n +
                0.05 * smoke_n +
                0.05 * alcohol_n +
                0.075 * (1 - exercise) +
                0.075 * (1 - diet) +
                0.075 * (1 - sleep)
            )

            interactions = (
                0.25 * max(0.0, age_n - 0.35) * smoke_n +
                0.20 * max(0.0, sys_n - 0.45) * max(0.0, dia_n - 0.45) +
                0.20 * ecg_prob * max(0.0, sys_n - 0.4) +
                0.05 * sex * max(0.0, age_n - 0.5)  # female age interaction (subtle)
            )

            raw_risk = base_risk + interactions
            raw_risk = max(0.0, min(raw_risk, 1.0))
            risk = raw_risk ** 1.7

            features = torch.tensor([
                ecg_prob,
                exercise, diet, sleep,
                smoke_n, alcohol_n,
                age_n,
                sex,
                bmi_n,
                sys_n, dia_n
            ], dtype=torch.float32)

            self.X.append(features)
            self.y.append(torch.tensor(risk, dtype=torch.float32))

        self.X = torch.stack(self.X).to(device)
        self.y = torch.stack(self.y).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


