import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(channels // reduction, 1), bias=False)
        self.fc2 = nn.Linear(max(channels // reduction, 1), channels, bias=False)

    def forward(self, x):
        # x: [B, C, L]
        b, c, _ = x.size()
        y = F.adaptive_avg_pool1d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, downsample=None, dropout=0.2):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class Net(nn.Module):
    """
    Integrated ECG model:
      - Stacked ResidualBlock1D encoder with SE
      - Temporal Multi-Head Attention (batch_first)
      - Global pooling + classifier
    """
    def __init__(self,
                 input_channels=1,
                 num_classes=1,
                 block_channels=(64, 128, 256),
                 kernel_sizes=(7,5,3),
                 strides=(2,2,2),
                 attention_heads=4,
                 transformer_dropout=0.2,
                 encoder_dropout=0.2,
                 classifier_dropout=0.3):
        super().__init__()

        assert len(block_channels) == len(kernel_sizes) == len(strides), "block configs must match"

        # initial conv (keeps things stable)
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, block_channels[0], kernel_size=kernel_sizes[0], stride=1,
                      padding=kernel_sizes[0]//2, bias=False),
            nn.BatchNorm1d(block_channels[0]),
            nn.ReLU(inplace=True)
        )

        # build residual stack
        layers = []
        in_ch = block_channels[0]
        for i, out_ch in enumerate(block_channels):
            k = kernel_sizes[i]
            s = strides[i]
            if in_ch != out_ch or s != 1:
                downsample = nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=s, bias=False),
                    nn.BatchNorm1d(out_ch)
                )
            else:
                downsample = None
            layers.append(ResidualBlock1D(in_channels=in_ch, out_channels=out_ch,
                                          kernel_size=k, stride=s, downsample=downsample, dropout=encoder_dropout))
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)

        # attention expects embed_dim == last channel size
        self.embed_dim = block_channels[-1]
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim,
                                               num_heads=attention_heads,
                                               dropout=transformer_dropout,
                                               batch_first=True)

        # small feed-forward after attention (residual)
        self.attn_ff = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(transformer_dropout),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        self.attn_norm = nn.LayerNorm(self.embed_dim)

        # classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(self.embed_dim // 2, num_classes)
        )

        # init weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.Linear, nn.Conv1d)):
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, C, L)
        x = self.stem(x)               # -> (B, block_channels[0], L)
        x = self.encoder(x)           # -> (B, embed_dim, L')

        # prepare for attention: (B, Seq, Features)
        x_seq = x.permute(0, 2, 1)    # (B, L', embed_dim)

        # self-attention + residual
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)  # (B, L', embed_dim)
        x_seq = x_seq + attn_out
        # FFN + norm
        ff = self.attn_ff(x_seq)
        x_seq = self.attn_norm(x_seq + ff)

        # back to (B, embed_dim, L')
        x = x_seq.permute(0, 2, 1)

        # global pooling + classifier
        pooled = self.global_pool(x).squeeze(-1)  # (B, embed_dim)
        out = self.classifier(pooled)             # (B, num_classes)
        return out
    


