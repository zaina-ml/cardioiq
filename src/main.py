import os
import torch
from config import CFG
from model import ECGNet
from data import ECGDataset, train_test_split, build_dataset
from train import train_model

if __name__ == "__main__":
    cfg = CFG
    print("Using device:", cfg.device)
    print("[INFO] Configuration: ", cfg)

    print("[INFO] Loading and preparing dataset...")
    X, y = build_dataset(cfg)
    X_train, y_train, X_val, y_val, X_test, y_test = train_test_split(X, y, cfg)

    print("[INFO] Initializing model...")
    model = ECGNet(input_len=cfg.crop_len, dropout_p=0.3)

    print("[INFO] Starting training...")
    trained_model, threshold, best_pr = train_model(
        model, X_train, y_train, X_val, y_val, cfg
    )

    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "cardioiq_model_modular.pt")
    torch.save({
        "model_state_dict": trained_model.state_dict(),
        "threshold": threshold,
        "best_pr": best_pr,
        "config": cfg.__dict__,
    }, save_path)

    print(f"[INFO] Model saved to {save_path}")