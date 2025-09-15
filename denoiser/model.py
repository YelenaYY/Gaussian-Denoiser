import torch
import torch.nn as nn
import torch.nn.init as init
from pathlib import Path
import warnings


class DnCNN(nn.Module):
    def __init__(self, num_layers=17, num_channels=64, kernel_size=3, image_channels=1, padding=1):
        super(DnCNN, self).__init__()
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=num_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers-2):
            layers.append(nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(num_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=num_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=True))
        self.dncnn = nn.Sequential(*layers)

        # Initialize weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.orthogonal_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)

    def forward(self, x):
        y = x
        noise = self.dncnn(x)
        return y - noise # cleaned image

def save_checkpoint(model, epoch, model_dir: Path):
    if not model_dir.exists():
        model_dir.mkdir(parents=True)
    model_path = model_dir / f'model_{epoch:03d}.pth'
    if not model_path.exists():
        torch.save(model.state_dict(), model_path)
    else:
        model_path = str(model_path) + '.new'
        warnings.warn(f"Model already exists at {model_path}, saving as {model_path}")
        torch.save(model.state_dict(), model_path)

def load_checkpoint(model, model_dir):
    checkpoints = list(model_dir.glob('model_*.pth'))
    if len(checkpoints) == 0:
        print("No checkpoints found, train from beginning")
        return
    # sort the checkpoints by the epoch number
    checkpoints = sorted(checkpoints, key=lambda x: int(x.stem.split('_')[-1]), reverse=True)

    # get the latest checkpoint
    latest_checkpoint = checkpoints[0]
    model.load_state_dict(torch.load(latest_checkpoint))
    print(f"Loaded checkpoint from {latest_checkpoint}")
    return True