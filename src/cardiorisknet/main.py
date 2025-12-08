import torch
import os

from config import CFG
from data import SyntheticPatientDataset
from model import CardioRiskNet
from train import train_model

print(f"[INFO] Creating synthetic dataset...")
dataset = SyntheticPatientDataset(num_samples=1000)

print(f"[INFO] Creating model...")
model = CardioRiskNet(input_size=dataset.features.shape[1]).to(CFG.device)

print(f"[INFO] Beginning training...")
model = train_model(model, dataset)

save_dir = "models"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "cardiorisknet_modular.pt")
torch.save({
    "model_state_dict": model.state_dict(),
}, save_path)

print(f"[INFO] Model saved to {save_path}")