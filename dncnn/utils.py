# utils.py - Helpers (PSNR, SSIM, file listing, etc.)
import os
import math
from typing import List
import numpy as np
import torch
from PIL import Image


def make_filelist(root_dir: str, exts=(".png", ".jpg", ".jpeg", ".bmp")) -> List[str]:
    """Recursively collect image files from a directory."""
    files = []
    for r, _, fns in os.walk(root_dir):
        for f in fns:
            if f.lower().endswith(exts):
                files.append(os.path.join(r, f))
    return files


def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse.item()))


def rgb_to_ycbcr(img: torch.Tensor) -> torch.Tensor:
    # Convert RGB to YCbCr color space.
    
    if img.dim() == 4:  # Batch of images
        img = img.squeeze(0)
    
    # Convert to numpy for easier manipulation
    img_np = img.permute(1, 2, 0).cpu().numpy()
    
    # Conversion matrix
    transform_matrix = np.array([[0.299, 0.587, 0.114],
                                [-0.168736, -0.331264, 0.5],
                                [0.5, -0.418688, -0.081312]])
    
    ycbcr = img_np @ transform_matrix.T
    ycbcr[:, :, 1:] += 0.5  # Add offset for Cb and Cr channels
    
    return torch.from_numpy(ycbcr).permute(2, 0, 1).to(img.device)


def calculate_psnr_ycbcr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate PSNR on Y channel of YCbCr color space."""
    if img1.size(0) == 3:  # RGB image
        img1_y = rgb_to_ycbcr(img1)[0:1]  # Y channel only
        img2_y = rgb_to_ycbcr(img2)[0:1]
        return psnr(img1_y, img2_y)
    else:  # Grayscale
        return psnr(img1, img2)


def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor in [0,1] to uint8 numpy array."""
    return (tensor.clamp(0, 1) * 255).byte().cpu().numpy()


def save_image(tensor: torch.Tensor, path: str):
    """Save tensor as image file."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    if tensor.size(0) == 1:  # Grayscale
        img_array = tensor_to_uint8(tensor.squeeze(0))
        img = Image.fromarray(img_array, mode='L')
    else:  # RGB
        img_array = tensor_to_uint8(tensor.permute(1, 2, 0))
        img = Image.fromarray(img_array, mode='RGB')
    
    img.save(path)