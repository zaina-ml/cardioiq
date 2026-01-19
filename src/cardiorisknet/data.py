import torch
from torch.utils.data import Dataset
from config import CFG
import random

def sample_regime():
    regimes = [
        "healthy",
        "lifestyle",
        "hypertensive",
        "cardiopulmonary",
        "high_risk"
    ]
    probs = [0.30, 0.25, 0.20, 0.15, 0.10]
    return random.choices(regimes, probs)[0]

def sample_patient_by_regime(regime):
    if regime == "healthy":
        age = random.uniform(20, 40)
        bmi = random.uniform(18.5, 24)
        sys_bp = random.uniform(90, 120)
        dia_bp = random.uniform(60, 80)
        heart_rate = random.uniform(55, 80)
        spo2 = random.uniform(97, 100)

        exercise = random.uniform(0.7, 1.0)
        diet = random.uniform(0.7, 1.0)
        sleep = random.uniform(0.7, 1.0)

        smoking = random.uniform(0, 10)
        alcohol = random.uniform(0, 7)
        ecg_prob = random.uniform(0.0, 0.2)

    elif regime == "lifestyle":
        age = random.uniform(25, 55)
        bmi = random.uniform(24, 30)
        sys_bp = random.uniform(110, 140)
        dia_bp = random.uniform(70, 90)
        heart_rate = random.uniform(65, 95)
        spo2 = random.uniform(95, 99)

        exercise = random.uniform(0.2, 0.6)
        diet = random.uniform(0.2, 0.6)
        sleep = random.uniform(0.3, 0.6)

        smoking = random.uniform(20, 80)
        alcohol = random.uniform(5, 20)
        ecg_prob = random.uniform(0.1, 0.4)

    elif regime == "hypertensive":
        age = random.uniform(45, 75)
        bmi = random.uniform(26, 35)
        sys_bp = random.uniform(140, 180)
        dia_bp = random.uniform(85, 120)
        heart_rate = random.uniform(70, 105)
        spo2 = random.uniform(94, 98)

        exercise = random.uniform(0.2, 0.5)
        diet = random.uniform(0.3, 0.6)
        sleep = random.uniform(0.4, 0.7)

        smoking = random.uniform(10, 60)
        alcohol = random.uniform(5, 15)
        ecg_prob = random.uniform(0.3, 0.6)

    elif regime == "cardiopulmonary":
        age = random.uniform(50, 80)
        bmi = random.uniform(22, 32)
        sys_bp = random.uniform(120, 170)
        dia_bp = random.uniform(70, 100)
        heart_rate = random.uniform(80, 110)
        spo2 = random.uniform(88, 95)

        exercise = random.uniform(0.1, 0.4)
        diet = random.uniform(0.3, 0.6)
        sleep = random.uniform(0.3, 0.6)

        smoking = random.uniform(30, 100)
        alcohol = random.uniform(5, 20)
        ecg_prob = random.uniform(0.4, 0.8)

    else:  # high_risk
        age = random.uniform(60, 85)
        bmi = random.uniform(28, 40)
        sys_bp = random.uniform(150, 200)
        dia_bp = random.uniform(90, 130)
        heart_rate = random.uniform(90, 130)
        spo2 = random.uniform(85, 92)

        exercise = random.uniform(0.0, 0.3)
        diet = random.uniform(0.1, 0.4)
        sleep = random.uniform(0.2, 0.5)

        smoking = random.uniform(60, 140)
        alcohol = random.uniform(10, 30)
        ecg_prob = random.uniform(0.6, 1.0)

    sex = random.choice([0, 1])

    return {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "sys_bp": sys_bp,
        "dia_bp": dia_bp,
        "heart_rate": heart_rate,
        "spo2": spo2,
        "exercise": exercise,
        "diet": diet,
        "sleep": sleep,
        "smoking": smoking,
        "alcohol": alcohol,
        "ecg_prob": ecg_prob
    }

class SyntheticPatientDataset(Dataset):
    def __init__(self, num_samples=2000, cfg=CFG):
        self.features = []
        self.targets = []

        for _ in range(num_samples):
            regime = sample_regime()
            p = sample_patient_by_regime(regime)

            age_norm = (p["age"] - 20) / 65
            bmi_norm = (p["bmi"] - 18.5) / 21.5
            sys_bp_norm = (p["sys_bp"] - 90) / 110
            dia_bp_norm = (p["dia_bp"] - 60) / 70
            smoking_norm = p["smoking"] / 140
            alcohol_norm = p["alcohol"] / 30
            hr_norm = min(abs(p["heart_rate"] - 75) / 40, 1.0)
            spo2_norm = min((100 - p["spo2"]) / 15, 1.0)

            feat = [
                p["ecg_prob"],
                p["exercise"],
                p["diet"],
                p["sleep"],
                smoking_norm,
                alcohol_norm,
                age_norm,
                p["sex"],
                bmi_norm,
                sys_bp_norm,
                dia_bp_norm,
                hr_norm,
                spo2_norm
            ]

            self.features.append(torch.tensor(feat, dtype=torch.float32))

            baseline_risk = (
                0.30 * age_norm +
                0.10 * p["sex"] +
                0.15 * bmi_norm
            )

            vitals_risk = (
                0.20 * sys_bp_norm +
                0.15 * dia_bp_norm +
                0.15 * hr_norm +
                0.25 * spo2_norm
            )

            lifestyle_risk = (
                0.20 * smoking_norm +
                0.05 * alcohol_norm +
                0.15 * (1 - p["exercise"]) +
                0.10 * (1 - p["diet"]) +
                0.10 * (1 - p["sleep"])
            )

            ecg_risk = 0.35 * p["ecg_prob"]

            risk = (
                0.30 * baseline_risk +
                0.35 * vitals_risk +
                0.20 * lifestyle_risk +
                ecg_risk
            )

            risk = min(max(risk, 0.0), 1.0)
            self.targets.append(torch.tensor(risk, dtype=torch.float32))

        self.features = torch.stack(self.features).to(cfg.device)
        self.targets = torch.stack(self.targets).to(cfg.device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]