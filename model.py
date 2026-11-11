"""CNN architecture adapted from cnn_mine for five band populations."""

import torch
import torch.nn as nn


class PhysicalResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.conv(inputs)
        batch, channels, _, _ = inputs.size()
        weights = self.se(residual).view(batch, channels, 1, 1)
        return self.relu(inputs + residual * weights)


class AdvancedLatticeCNN(nn.Module):
    """Original cnn_mine backbone with only five orbit outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            PhysicalResidualBlock(64),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            PhysicalResidualBlock(128),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            PhysicalResidualBlock(256),
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc_shared = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        self.orbit_head = nn.Linear(128, 5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stage1(inputs)
        features = self.stage2(features)
        features = self.stage3(features)
        features = self.stage4(features)
        features = self.pool(features)
        features = torch.flatten(features, 1)
        features = self.fc_shared(features)

        orbits = torch.softmax(self.orbit_head(features), dim=1)
        return orbits
