"""
Author: Yue (Yelena) Yu,  Rongfei (Eric) JIn
Date: 2025-09-23
Class: CS 7180 Advanced Perception
"""
import os
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

from dncnn.models import DnCNN
from dncnn.utils import rgb_to_ycbcr, psnr, make_filelist


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, device: str = "cpu") -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d


def ssim_y(img1_y: torch.Tensor, img2_y: torch.Tensor, max_val: float = 1.0) -> float:
    """Compute SSIM on Y (luminance) channel. Inputs are 1xHxW tensors in [0,1]."""
    if img1_y.dim() != 3 or img2_y.dim() != 3:
        raise ValueError("ssim_y expects 1xHxW tensors")

    device = img1_y.device
    window_size = 11
    sigma = 1.5
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    kernel = _gaussian_kernel(window_size, sigma, device=device).unsqueeze(0).unsqueeze(0)

    def _filter(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x.unsqueeze(0), kernel, padding=window_size // 2).squeeze(0)

    x = img1_y.unsqueeze(0)  # 1x1xHxW for conv2d
    y = img2_y.unsqueeze(0)

    mu_x = F.conv2d(x, kernel, padding=window_size // 2)
    mu_y = F.conv2d(y, kernel, padding=window_size // 2)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, kernel, padding=window_size // 2) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, kernel, padding=window_size // 2) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, padding=window_size // 2) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
    return ssim_map.mean().item()


@torch.inference_mode()
def _prepare_y_channels(hr_img: Image.Image, scale: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given an HR PIL image, create bicubic-downscaled-and-upscaled image and return (HR_Y, ILR_Y) tensors.
    Returns tensors of shape 1xHxW in [0,1] on the specified device.
    """
    hr_w, hr_h = hr_img.size
    lr_w, lr_h = max(1, hr_w // scale), max(1, hr_h // scale)

    # Bicubic downscale then upscale
    lr = hr_img.resize((lr_w, lr_h), Image.BICUBIC)
    ilr = lr.resize((hr_w, hr_h), Image.BICUBIC)

    # To tensors (C,H,W) in [0,1]
    hr_t = to_tensor(hr_img).to(device)
    ilr_t = to_tensor(ilr).to(device)

    # Convert to Y channel
    hr_y = rgb_to_ycbcr(hr_t)[0:1]
    ilr_y = rgb_to_ycbcr(ilr_t)[0:1]
    return hr_y, ilr_y


@torch.inference_mode()
def evaluate_model_on_sisr(model_path: str, depth: int, dataset_dirs: List[str], scales: Tuple[int, ...]) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DnCNN(depth=depth, in_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = {}

    for ds_dir in dataset_dirs:
        files = sorted([f for f in make_filelist(ds_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))])
        if not files:
            continue

        ds_name = os.path.basename(ds_dir.rstrip("/"))
        results[ds_name] = {}

        for s in scales:
            psnrs: List[float] = []
            ssims: List[float] = []

            for fpath in files:
                hr = Image.open(fpath).convert("RGB")
                hr_y, ilr_y = _prepare_y_channels(hr, s, device)

                # Model expects 1-channel input
                denoised_y, _ = model(ilr_y.unsqueeze(0))  # 1x1xHxW
                denoised_y = denoised_y.squeeze(0)

                psnr_val = psnr(hr_y, denoised_y)
                ssim_val = ssim_y(hr_y, denoised_y)

                psnrs.append(psnr_val)
                ssims.append(ssim_val)

            results[ds_name][f"x{s}"] = {
                "avg_psnr": sum(psnrs) / len(psnrs),
                "avg_ssim": sum(ssims) / len(ssims),
                "count": len(psnrs),
            }

    return results


def _print_results(title: str, metrics: Dict):
    print(f"\n=== {title} ===")
    for ds_name, scales_dict in metrics.items():
        print(f"Dataset: {ds_name}")
        for scale, vals in scales_dict.items():
            print(f"  Scale {scale}: PSNR {vals['avg_psnr']:.2f} dB | SSIM {vals['avg_ssim']:.4f} | N={vals['count']}")


def main():
    p = argparse.ArgumentParser(description="Evaluate average PSNR/SSIM on SISR (Y channel) as in DnCNN paper")
    p.add_argument("--model_s", type=str, default=os.path.join("Training results", "dncnn_S_sigma25_50epochs.pt"),
                   help="Path to 50-epoch DnCNN-S checkpoint")
    p.add_argument("--model_b", type=str, default=os.path.join("Training results", "dncnn_B_50epochs.pt"),
                   help="Path to 50-epoch DnCNN-B checkpoint")
    p.add_argument("--depth_s", type=int, default=17, help="Depth for DnCNN-S (paper: 17)")
    p.add_argument("--depth_b", type=int, default=20, help="Depth for DnCNN-B (paper: 20)")
    p.add_argument("--set5", type=str, default=os.path.join("data", "Set5", "Set5"), help="Set5 directory with HR images")
    p.add_argument("--set14", type=str, default=os.path.join("data", "Set14", "Set14"), help="Set14 directory with HR images")
    p.add_argument("--scales", type=int, nargs="+", default=[2, 3, 4], help="SISR scales to evaluate")
    args = p.parse_args()

    datasets = [args.set5, args.set14]

    s_metrics = evaluate_model_on_sisr(args.model_s, args.depth_s, datasets, tuple(args.scales))
    b_metrics = evaluate_model_on_sisr(args.model_b, args.depth_b, datasets, tuple(args.scales))

    _print_results("DnCNN-S (50 epochs)", s_metrics)
    _print_results("DnCNN-B (50 epochs)", b_metrics)


if __name__ == "__main__":
    main()


