import torch
from torch import nn

class ECGGenerator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256 * 45)

        self.net = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, 2, 1),
            nn.LayerNorm([128, 90]),
            nn.ReLU(),

            nn.ConvTranspose1d(128, 64, 4, 2, 1),
            nn.LayerNorm([64, 180]),
            nn.ReLU(),

            nn.ConvTranspose1d(64, 32, 4, 2, 1),
            nn.LayerNorm([32, 360]),
            nn.ReLU(),

            nn.ConvTranspose1d(32, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 45)
        return self.net(x)
