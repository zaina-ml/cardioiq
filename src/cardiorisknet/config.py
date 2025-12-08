import torch
from dataclasses import dataclass

@dataclass
class TrainCfg:
    batch_size: int = 128
    max_epochs: int = 150
    lr: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = TrainCfg()