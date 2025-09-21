# __init__.py - Package initialization
"""
DnCNN: Deep CNN for Image Denoising

This package provides implementations of DnCNN models for image denoising,
including training and inference utilities.
"""

from .models import DnCNN
from .datasets import PatchDataset, add_gaussian_noise, degrade_sisr, degrade_jpeg
from .utils import make_filelist, psnr, calculate_psnr_ycbcr, save_image
from .inference import denoise_image, evaluate_denoising, batch_denoise
from .train import TrainCfg, train_loop, train_dncnn_s, train_dncnn_b, train_dncnn_3

__version__ = "1.0.0"
__author__ = "DnCNN Implementation"

__all__ = [
    # Models
    "DnCNN",
    
    # Datasets and degradations
    "PatchDataset",
    "add_gaussian_noise",
    "degrade_sisr", 
    "degrade_jpeg",
    
    # Utilities
    "make_filelist",
    "psnr",
    "calculate_psnr_ycbcr",
    "save_image",
    
    # Inference
    "denoise_image",
    "evaluate_denoising",
    "batch_denoise",
    
    # Training
    "TrainCfg",
    "train_loop",
    "train_dncnn_s",
    "train_dncnn_b", 
    "train_dncnn_3",
]
