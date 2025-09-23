# Authors:Rongfei Jin and Yelena Yu,
# Date: 2025-09-23, 
# Course: CS 7180 Advanced Perception
from denoiser.dataset import (
    PatchDataset,
    DEFAULT_TRANSFORM,
    MODEL_S_NOISE_TRANSFORM,
    MODEL_B_NOISE_TRANSFORM,
    MODEL_3_NOISE_TRANSFORM,
)
from denoiser.model import DnCNN, load_latest_checkpoint, save_checkpoint
from denoiser.utils import Logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from tqdm import tqdm
from pathlib import Path
from datetime import datetime


def train(options: dict):
    train_data = (
        options["train_data"] if "train_data" in options else ["data/train/TRAIN400"]
    )
    model_dir = options["model_dir"] if "model_dir" in options else "models"
    max_epoch = options["max_epoch"] if "max_epoch" in options else 100
    log_dir = options["log_dir"] if "log_dir" in options else "logs"
    batch_size = options["batch_size"] if "batch_size" in options else 128
    checkpoint = options["checkpoint"] if "checkpoint" in options else None

    model_type = options["model_type"]
    if model_type == "s":
        patch_size = 40
        stride = 6
        noise_transform = MODEL_S_NOISE_TRANSFORM
        image_channels = 1
        num_layers = 17

    elif model_type == "cb":
        patch_size = 50
        stride = 11
        noise_transform = MODEL_B_NOISE_TRANSFORM
        image_channels = 3
        num_layers = 20
    elif model_type == "3":
        patch_size = 50
        stride = 5
        noise_transform = MODEL_3_NOISE_TRANSFORM
        image_channels = 3
        num_layers = 20
    else:
        raise ValueError("Invalid model type! must be one of s/b/3")

    model_dir = Path(model_dir) / model_type
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir) / model_type
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(
        log_dir / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        headers=["epoch", "batch_idx", "loss", "avg_loss"],
    )

    use_cuda = torch.cuda.is_available()

    model = DnCNN(
        num_layers=num_layers,
        image_channels=image_channels,
    )
    if checkpoint:
        if Path(checkpoint).exists():
            model.load_state_dict(torch.load(checkpoint))
        else:
            raise FileNotFoundError(f"Checkpoint {checkpoint} not found")
    else:
        load_latest_checkpoint(model, model_dir)

    model.train()
    if use_cuda:
        model = model.cuda()

    criterion = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = MultiStepLR(
        optimizer, milestones=[30, 60, 90], gamma=0.2
    )  # learning rates

    for epoch in range(max_epoch):
        epoch_loss = 0

        training_dataset = PatchDataset(
            train_data,
            transform=DEFAULT_TRANSFORM,
            patch_size=patch_size,
            stride=stride,
            batch_size=128,
            noise_transform=noise_transform,
        )
        dataloader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{max_epoch}", leave=False)
        for n_count, (noisy_patches, patches) in enumerate(pbar):
            optimizer.zero_grad()
            if use_cuda:
                noisy_patches = noisy_patches.cuda()
                patches = patches.cuda()

            output = model(noisy_patches)  # cleaned patches
            target = patches
            loss = criterion(output, target)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            # Update progress bar with current loss
            avg_loss = epoch_loss / (n_count + 1)
            pbar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Avg Loss": f"{avg_loss:.4f}"}
            )
            if n_count % 128 == 0:
                data = {
                    "epoch": epoch,
                    "batch_idx": n_count,
                    "loss": loss.item(),
                    "avg_loss": avg_loss,
                }
                logger.log(data)

        scheduler.step()
        save_checkpoint(model, epoch + 1, model_dir)
