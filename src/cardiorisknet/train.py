import torch
from torch import nn
from torch import optim

from torch.utils.data import DataLoader
from config import CFG
from model import CardioRiskNet
from data import SyntheticPatientDataset


def train_model(model, dataset, cfg=CFG):
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
    criterion = nn.MSELoss()

    for epoch in range(cfg.max_epochs):
        total_loss = 0
        model.train()
        for X, y in loader:
            optimizer.zero_grad()
            pred = model(X).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{cfg.max_epochs}, Loss: {total_loss/len(loader):.4f}")
    return model

