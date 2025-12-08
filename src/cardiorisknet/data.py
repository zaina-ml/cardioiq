import torch
from torch.utils.data import Dataset
import random
from config import CFG

class SyntheticPatientDataset(Dataset):
    def __init__(self, num_samples=2000, cfg=CFG):
        self.features = []
        self.targets = []

        for _ in range(num_samples):
            # Core features
            ecg_prob = random.uniform(0, 1)
            exercise = random.uniform(0, 1)
            diet = random.uniform(0, 1)
            sleep = random.uniform(0, 1)
            smoking = random.choice([0, 1])
            alcohol = random.choice([0, 1])
            age = random.uniform(20, 80)  # raw age
            bmi = random.uniform(18.5, 30)
            bp = random.uniform(90, 180)
            chol = random.uniform(150, 300)
            sex = random.choice([0, 1])  # male=0, female=1

            # Normalize all numeric features to 0–1
            age_norm = (age - 20)/60
            bmi_norm = (bmi - 18.5)/11.5
            bp_norm = (bp - 90)/90
            chol_norm = (chol - 150)/150

            feat = [
                ecg_prob, exercise, diet, sleep,
                smoking, alcohol, age_norm, sex,
                bmi_norm, bp_norm, chol_norm
            ]
            self.features.append(torch.tensor(feat, dtype=torch.float32))

            # Simple risk: ECG dominates, others add minor contributions
            risk = min(max(
                0.5 * ecg_prob + 
                0.1 * (1 - exercise) + 
                0.1 * (1 - diet) + 
                0.1 * (1 - sleep) + 
                0.05 * smoking + 
                0.05 * alcohol + 
                0.1 * age_norm + 
                0.05 * bmi_norm + 
                0.05 * bp_norm + 
                0.05 * chol_norm,
                0), 1
            )

            self.targets.append(torch.tensor(risk, dtype=torch.float32))

        self.features = torch.stack(self.features).to(cfg.device)
        self.targets = torch.stack(self.targets).to(cfg.device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
