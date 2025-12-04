from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TrainCfg:
    window_sec = 2.0
    crop_len = 360
    records = tuple([
        '100','101','102','103','104','105','106','107','108','109','111','112','113','114','115',
        '116','117','118','119','121','122','123','124','200','201','202','203','205','207','208',
        '209','210','212','213','214','215','217','219','220','221','222','223','228','230','231',
        '232','233','234'
    ])
    label_keep = ("N", "V", "F")
    split_mode = "stratified"
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    crop_len: int = 720
    batch_size: int = 128
    max_epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    use_sampler: bool = True
    sampler_scale: float = 0.2
    focal_gamma: float = 1.5
    hybrid_switch_epoch: int = 15
    augment: bool = True
    severity: int = 0.25
    mixup_p: float = 0.5
    mixup_alpha: float = 0.1
    clip_grad_norm: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    save_dir: str = "checkpoints"
    seed: int = 42

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=kernel//2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=kernel//2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.down = None
        if in_ch != out_ch or stride != 1:
            self.down = nn.Conv1d(in_ch, out_ch, 1, stride=stride)

    def forward(self, x):
        identity = x if self.down is None else self.down(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out

class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        w = x.mean(dim=2)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w)).unsqueeze(-1)
        return x * w


class MultiScaleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv3 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv5 = nn.Conv1d(in_ch, out_ch, 5, padding=2)
        self.conv7 = nn.Conv1d(in_ch, out_ch, 7, padding=3)
        self.bn = nn.BatchNorm1d(out_ch*3)

    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        out = torch.cat([x3, x5, x7], dim=1)
        return F.relu(self.bn(out))


class ECGNet(nn.Module):
    def __init__(self, input_len=720, dropout_p=0.5, temporal_context=False, lstm_hidden=256):
        super().__init__()
        self.temporal_context = temporal_context

        self.stem = nn.Conv1d(1, 64, 7, padding=3)

        self.r1 = ResidualBlock(64, 128)
        self.r2 = ResidualBlock(128, 256)
        self.r3 = ResidualBlock(256, 256)
        self.r4 = ResidualBlock(256, 512)
        self.se = SEBlock1D(512)

        self.ms = MultiScaleConv(512, 256)

        self.lstm = None
        if temporal_context:
            self.lstm = nn.LSTM(input_size=256*3, hidden_size=lstm_hidden,
                                batch_first=True, bidirectional=True)
            self.fc = nn.Linear(lstm_hidden*2, 1)
        else:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.maxpool = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Linear(256*3*2, 1)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.stem(x)
        x = self.r1(x)
        x = self.r2(x)
        x = self.r3(x)
        x = self.r4(x)
        x = self.se(x)
        x = self.ms(x)

        if self.temporal_context:
            B, T, C, L = x.shape
            x = x.view(B, T, -1)
            x, _ = self.lstm(x)
            x = x[:, -1, :]
        else:
            a = self.avgpool(x).squeeze(-1)
            m = self.maxpool(x).squeeze(-1)
            x = torch.cat([a, m], dim=1)

        x = self.dropout(x)
        return self.fc(x).squeeze(1)
