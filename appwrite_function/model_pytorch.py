"""Minimal model definition for the Appwrite function deployment."""

import torch.nn as nn
from torchvision import models


class BrainTumorResNet(nn.Module):
    def __init__(self, num_classes=4, pretrained=False, dropout_rate=0.5):
        super().__init__()

        weights = 'IMAGENET1K_V1' if pretrained else None
        self.base_model = models.resnet50(weights=weights)

        for param in self.base_model.parameters():
            param.requires_grad = False

        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.base_model(x)