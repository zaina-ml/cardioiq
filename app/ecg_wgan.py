import torch
from torch import nn

class ECGGenerator(nn.Module):
    def __init__(self, z_dim=100, n_classes=2):
        super().__init__()

        self.embed = nn.Embedding(n_classes, z_dim)
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

    def forward(self, z, y):
        y_emb = self.embed(y)
        z = z + y_emb
        x = self.fc(z).view(-1, 256, 45)
        out = self.net(x)

        out = torch.clamp(out, -0.999, 0.999)
        return out