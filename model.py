"""ResNet-8 for Fashion-MNIST (He et al. 2016, CIFAR variant with depth 6n+2, n=1)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Two 3x3 convs with a skip connection: out = ReLU(F(x) + shortcut(x))."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # bias=False: a BatchNorm follows directly and has its own shift (beta),
        # so a conv bias would be redundant.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If the block changes the spatial size (stride 2) or the channel count,
        # x can't be added to F(x) directly. We project it with a 1x1 conv
        # ("option B" in the paper). Otherwise the shortcut is the identity.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Note: the ReLU comes *after* the addition (original post-activation design).
        return F.relu(out + self.shortcut(x))


class ResNet8(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.0):
        super().__init__()
        # Stem: 1 x 28 x 28 -> 16 x 28 x 28
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = BasicBlock(16, 16, stride=1)  # -> 16 x 28 x 28
        self.stage2 = BasicBlock(16, 32, stride=2)  # -> 32 x 14 x 14
        self.stage3 = BasicBlock(32, 64, stride=2)  # -> 64 x 7 x 7
        self.pool = nn.AdaptiveAvgPool2d(1)  # global average pool -> 64 x 1 x 1
        # Dropout on the 64-dim feature vector, right before the classifier.
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(64, num_classes)

        # He/Kaiming init is designed for ReLU networks: it keeps the variance
        # of activations roughly constant from layer to layer.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = torch.flatten(self.pool(x), 1)  # (B, 64, 1, 1) -> (B, 64)
        x = self.dropout(x)
        return self.fc(x)  # raw logits; CrossEntropyLoss applies softmax itself


if __name__ == "__main__":
    # Sanity check: shapes per stage and parameter count.
    model = ResNet8(dropout=0.2)
    x = torch.randn(4, 1, 28, 28)
    for name in ["stem", "stage1", "stage2", "stage3"]:
        x = getattr(model, name)(x)
        print(f"{name:7s} {tuple(x.shape)}")
    print("logits ", tuple(model(torch.randn(4, 1, 28, 28)).shape))
    print("params ", sum(p.numel() for p in model.parameters()))
