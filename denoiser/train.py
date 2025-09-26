# Authors:Rongfei Jin and Yelena Yu,
# Date: 2025-09-23,
# Course: CS 7180 Advanced Perception
# File Description:
# This file contains the training code for the denoiser model.
# It includes the train function, which is used to train the denoiser model.
# The train function is used to train the denoiser model.

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
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm
from pathlib import Path
from datetime import datetime


def train(options: dict):
    train_data = options["train_data"] if "train_data" in options else ["data/train/TRAIN400"]
    model_dir = options["model_dir"] if "model_dir" in options else "models"
    max_epoch = options["max_epoch"] if "max_epoch" in options else 100
    log_dir = options["log_dir"] if "log_dir" in options else "logs"
    batch_size = options["batch_size"] if "batch_size" in options else 128
    checkpoint = options["checkpoint"] if "checkpoint" in options else None

    model_type = options["model_type"]

    # patch size and stride are pre-determined to match the model total patch number
    # since the dataset is different for each model, the stride can be different for each model
    if model_type == "s":
        patch_size = 40
        stride = 6
        noise_transform = MODEL_S_NOISE_TRANSFORM
        image_channels = 1
        num_layers = 17
        num_patches_per_batch = 1600
    elif model_type == "b":
        patch_size = 50
        stride = 4
        noise_transform = MODEL_B_NOISE_TRANSFORM
        image_channels = 1
        num_layers = 20
        num_patches_per_batch = 3000
    elif model_type == "cb":
        patch_size = 50
        stride = 11
        noise_transform = MODEL_B_NOISE_TRANSFORM
        image_channels = 3
        num_layers = 20
        num_patches_per_batch = 3000
    elif model_type == "3":
        patch_size = 50
        stride = 5
        noise_transform = MODEL_3_NOISE_TRANSFORM
        image_channels = 3
        num_layers = 20
        num_patches_per_batch = 8000
    else:
        raise ValueError("Invalid model type! must be one of s/b/cb/3")

    model_dir = Path(model_dir) / model_type
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir) / model_type
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(
        log_dir / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        headers=["epoch", "batch_idx", "loss", "avg_loss", "psnr_in", "psnr_out"],
    )

    use_cuda = torch.cuda.is_available()

    model = DnCNN(
        num_layers=num_layers,
        image_channels=image_channels,
    )

    # Load checkpoint if need to resume training
    if checkpoint:
        if Path(checkpoint).exists():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(torch.load(checkpoint, map_location=device))
        else:
            raise FileNotFoundError(f"Checkpoint {checkpoint} not found")
    else:
        load_latest_checkpoint(model, model_dir)

    
    model.train()

    if use_cuda:
        model = model.cuda()

    criterion = nn.MSELoss(reduction="sum")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # adaptive learning rate
    scheduler = ExponentialLR(
        optimizer,
        gamma=(1e-4 / 1e-3) ** (1 / 50),  # decay from 1e-3 to 1e-4 over 50 epochs
    )

    training_dataset = PatchDataset(
        train_data,
        transform=DEFAULT_TRANSFORM,
        patch_size=patch_size,
        stride=stride,
        batch_size=128,
        num_patches_per_batch=num_patches_per_batch,
        noise_transform=noise_transform,
    )
    dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    for epoch in range(max_epoch):
        epoch_loss = 0
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

            with torch.no_grad():
                batch_size = noisy_patches.size(0)
                psnr_in = 0
                psnr_out = 0
                for i in range(batch_size):
                    psnr_in += peak_signal_noise_ratio(patches[i].cpu().numpy(), noisy_patches[i].cpu().numpy())
                    psnr_out += peak_signal_noise_ratio(patches[i].cpu().numpy(), output[i].cpu().numpy())
                psnr_in /= batch_size
                psnr_out /= batch_size

            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Avg Loss": f"{avg_loss:.4f}", "PSNR In": f"{psnr_in:.4f}", "PSNR Out": f"{psnr_out:.4f}"})
            if n_count % 128 == 0:
                data = {
                    "epoch": epoch,
                    "batch_idx": n_count,
                    "loss": loss.item(),
                    "avg_loss": avg_loss,
                    "psnr_in": psnr_in,
                    "psnr_out": psnr_out,
                }
                logger.log(data)

        scheduler.step()
        save_checkpoint(model, epoch + 1, model_dir)
