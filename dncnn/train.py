"""
Author: Yue (Yelena) Yu,  Rongfei (Eric) JIn
Date: 2025-09-23
Class: CS 7180 Advanced Perception
"""
# train.py - Training loop + CLI
import argparse
import os
import csv
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import DnCNN
from datasets import PatchDataset
from utils import make_filelist, psnr


@dataclass
class TrainCfg:
    variant: str            # 'S', 'B', or '3'
    depth: int
    in_channels: int
    epochs: int = 30
    batch_size: int = 128
    lr_start: float = 1e-1          # starting LR
    lr_end: float = 1e-4            # ending LR
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "dncnn.pt"


def exp_lr(epoch, total, lr0, lrf):
    t = epoch / (total-1)
    return lr0 * (lrf / lr0) ** t


def train_loop(model, loader, cfg: TrainCfg):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr_start, momentum=0.9, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()

    # Prepare logging (CSV in project root)
    csv_path = os.path.join(".", "metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["epoch", "lr", "loss", "psnr_in", "psnr_out"]) 

    for ep in range(cfg.epochs):
        # update LR
        for g in opt.param_groups:
            g['lr'] = exp_lr(ep, cfg.epochs, cfg.lr_start, cfg.lr_end)

        running = 0.0
        num_batches = 0
        running_psnr_in = 0.0
        running_psnr_out = 0.0

        for y, x, v in loader:
            y, x, v = y.to(cfg.device), x.to(cfg.device), v.to(cfg.device)
            
            x_hat, v_hat = model(y)
            loss = mse(v_hat, v)  # residual loss
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            opt.step()
            running += loss.item() * y.size(0)
            num_batches += 1

            # Batch PSNRs (aggregated over batch elements)
            with torch.no_grad():
                running_psnr_in += psnr(x, y) * y.size(0)
                running_psnr_out += psnr(x, x_hat) * y.size(0)

        total_samples = num_batches * cfg.batch_size
        avg_loss = running / total_samples if total_samples > 0 else float('nan')
        avg_psnr_in = running_psnr_in / total_samples if total_samples > 0 else float('nan')
        avg_psnr_out = running_psnr_out / total_samples if total_samples > 0 else float('nan')
        print(f"Epoch {ep+1:03d}/{cfg.epochs} | LR {opt.param_groups[0]['lr']:.2e} | "
            f"MSE {avg_loss:.6f} | PSNR_in {avg_psnr_in:.2f} | PSNR_out {avg_psnr_out:.2f} | Batches: {num_batches}")

        # CSV logging
        with open(csv_path, mode='a', newline='') as f:
            w = csv.writer(f)
            w.writerow([ep + 1, f"{opt.param_groups[0]['lr']:.8f}", f"{avg_loss:.6f}", f"{avg_psnr_in:.4f}", f"{avg_psnr_out:.4f}"])

    torch.save(model.state_dict(), cfg.save_path)
    print(f"Saved: {cfg.save_path}")
    # No writer to close


# ---------- Training ----------

def train_dncnn_s(train_root: str, sigma: float, grayscale=True, save="dncnn_S_sigma{}.pt"):
    print(f"Loading training files from: {train_root}")
    files = make_filelist(train_root)
    print(f"Found {len(files)} training files")
    # Paper specification: 128 × 1,600 = 204,800 patches for DnCNN-S
    ds = PatchDataset(files, patch_size=40, grayscale=grayscale,
                      task="gauss", fixed_sigma=sigma, total_patches=128 * 1600)
    print(f"Created dataset with {len(ds)} samples (DnCNN-S: 204,800 patches)")
    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    depth = 17                        # paper: 17 for specific σ (≈35x35 RF)
    in_ch = 1 if grayscale else 3
    cfg = TrainCfg(variant='S', depth=depth, in_channels=in_ch, save_path=save.format(int(sigma)))
    print(f"Using device: {cfg.device}")
    model = DnCNN(depth=cfg.depth, in_channels=cfg.in_channels).to(cfg.device)
    print("Starting training...")
    train_loop(model, dl, cfg)


def train_dncnn_b(train_root: str, grayscale=True, save="dncnn_B.pt"):
    print(f"Loading training files from: {train_root}")
    files = make_filelist(train_root)
    print(f"Found {len(files)} training files")
    # Paper specification: 128 × 3,000 = 384,000 patches for DnCNN-B
    ds = PatchDataset(files, patch_size=50, grayscale=grayscale,
                      task="gauss", sigma_range=(0, 55), total_patches=128 * 3000)
    print(f"Created dataset with {len(ds)} samples (DnCNN-B: 384,000 patches)")
    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    depth = 20                        # paper: 20 for blind/general
    in_ch = 1 if grayscale else 3
    cfg = TrainCfg(variant='B', depth=depth, in_channels=in_ch, save_path=save)
    print(f"Using device: {cfg.device}")
    model = DnCNN(depth=cfg.depth, in_channels=cfg.in_channels).to(cfg.device)
    print("Starting training...")
    train_loop(model, dl, cfg)


def train_dncnn_3(train_root: str, save="dncnn_3.pt"):
    print(f"Loading training files from: {train_root}")
    files = make_filelist(train_root)
    print(f"Found {len(files)} training files")
    # Multi-task: use same patch count as DnCNN-B (no specific count given in paper)
    ds = PatchDataset(files, patch_size=50, grayscale=True,
                      task="multi", sigma_range=(0, 55),
                      sisr_scales=(2, 3, 4), jpeg_q_range=(5, 99), total_patches=128 * 3000)
    print(f"Created dataset with {len(ds)} samples (DnCNN-3: 384,000 patches)")
    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    cfg = TrainCfg(variant='3', depth=20, in_channels=1, save_path=save)
    print(f"Using device: {cfg.device}")
    model = DnCNN(depth=cfg.depth, in_channels=cfg.in_channels).to(cfg.device)
    print("Starting training...")
    train_loop(model, dl, cfg)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train DnCNN variants")
    p.add_argument("--variant", choices=["S", "B", "3"], required=True)
    p.add_argument("--train_root", required=True, help="Folder with clean training images")
    p.add_argument("--sigma", type=float, default=25.0, help="Used for DnCNN-S")
    p.add_argument("--save", type=str, default=None)
    args = p.parse_args()

    if args.variant == "S":
        save = args.save or f"dncnn_S_sigma{int(args.sigma)}.pt"
        train_dncnn_s(args.train_root, sigma=args.sigma, save=save)
    elif args.variant == "B":
        save = args.save or "dncnn_B.pt"
        train_dncnn_b(args.train_root, save=save)
    else:
        save = args.save or "dncnn_3.pt"
        train_dncnn_3(args.train_root, save=save)