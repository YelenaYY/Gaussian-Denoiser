"""
Author: Yue (Yelena) Yu,  Rongfei (Eric) JIn
Date: 2025-09-23
Class: CS 7180 Advanced Perception
"""
import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """
    DnCNN: Conv + ReLU (first), [Conv + BN + ReLU]x(D-2), Conv (last).
    Outputs: the residual (noise/error) R(y); 
    clean = y - R(y).
    """
    def __init__(self, depth: int = 17, in_channels: int = 1, features: int = 64):
        super().__init__()
        layers = []

        # first layer: Conv + ReLU
        layers += [nn.Conv2d(in_channels, features, 3, padding=1, bias=True), nn.ReLU(inplace=True)]

        # middle layers: Conv + BN + ReLU
        for _ in range(depth - 2):
            layers += [nn.Conv2d(features, features, 3, padding=1, bias=False),
                       nn.BatchNorm2d(features, eps=1e-5, momentum=0.1),
                       nn.ReLU(inplace=True)]

        # last layer: Conv
        layers += [nn.Conv2d(features, in_channels, 3, padding=1, bias=True)]
        self.net = nn.Sequential(*layers)
        self._init_kaiming()

    def _init_kaiming(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, y):
        v_hat = self.net(y)  # predict residual/noise
        x_hat = y - v_hat
        return x_hat, v_hat