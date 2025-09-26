# Authors:Rongfei Jin and Yelena Yu,
# Date: 2025-09-23, 
# Course: CS 7180 Advanced Perception
# File Description:
# This file contains the dataset class for the denoiser model.
# It includes the patch dataset class, the noise transform class, and the default transform class.
# The patch dataset class is used to load the images and extract the patches.
# The noise transform class is used to add noise to the patches.
# The default transform class is used to transform the patches.

from typing import Any
import random

import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset
from denoiser.utils import decode_any_image, load_images


# This is a transformation class that downsamples an image by a random factor and then upsamples it back to the original size
class BicubicDownThenUp:
    def __init__(self, scale_factors=[2, 3, 4]):
        self.scale_factors = scale_factors

    def __call__(self, img: torch.Tensor):
        original_size = img.shape[-2:]
        scale_factor = random.choice(self.scale_factors)

        down_h = max(1, original_size[0] // scale_factor)
        down_w = max(1, original_size[1] // scale_factor)

        downsampled = v2.functional.resize(
            img,
            size=list((down_h, down_w)),
            interpolation=v2.InterpolationMode.BICUBIC,
            antialias=True,
        )

        upsampled = v2.functional.resize(
            downsampled,
            size=list(original_size),
            interpolation=v2.InterpolationMode.BICUBIC,
            antialias=True,
        )

        return upsampled


# This is a transformation class that adds Gaussian noise to an image with a random sigma, 
# The sigma needs to scale with the image datatype, e.g. for a sigma 50, the noise passed to GaussianNoise should be 50/255.0
class RandomSigmaGaussianNoise:
    def __init__(self, noise: tuple[float, float] | float) -> None:
        if isinstance(noise, tuple):
            self.sigma = random.uniform(noise[0], noise[1])
        else:
            self.sigma = noise

    def __call__(self, img: torch.Tensor):
        self.transform = v2.GaussianNoise(sigma=self.sigma)
        return self.transform(img)


# This is a transformation class that applies JPEG noise to an image with a random quality
class FloatJPEG:
    def __init__(self, quality: list[int]):
        self.quality = quality

    def __call__(self, img: torch.Tensor):
        quality = random.choice(self.quality)
        return v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.JPEG(quality),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )(img)

def DEFAULT_TRANSFORM():
    return v2.RandomChoice(
    [
        v2.RandomRotation((-90, -90)),
        v2.RandomRotation((-180, -180)),
        v2.RandomRotation((-270, -270)),
        v2.RandomVerticalFlip(1),
        v2.Compose(
            [
                v2.RandomRotation((-90, -90)),
                v2.RandomVerticalFlip(1),
            ]
        ),
        v2.Compose(
            [
                v2.RandomRotation((-180, -180)),
                v2.RandomVerticalFlip(1),
            ]
        ),
        v2.Compose(
            [
                v2.RandomRotation((-270, -270)),
                v2.RandomVerticalFlip(1),
            ]
        ),
    ]
)

# The following are the noise transforms for the three models

def MODEL_S_NOISE_TRANSFORM():
    return RandomSigmaGaussianNoise(25.0 / 255.0)

def MODEL_B_NOISE_TRANSFORM():
    return RandomSigmaGaussianNoise((0, 55.0 / 255.0))

def MODEL_3_NOISE_TRANSFORM():
    return v2.RandomChoice(
    [
        FloatJPEG([10, 20, 30, 40]),
        BicubicDownThenUp([2, 3, 4]),
        RandomSigmaGaussianNoise((0, 55.0 / 255.0)),
    ]
)


class PatchDataset(Dataset):
    def __init__(
        self,
        data_dir: str | list[str],
        patch_size,
        stride,
        batch_size,
        num_patches_per_batch,
        transform=None,
        noise_transform: Any = MODEL_B_NOISE_TRANSFORM(),
    ):
        self.image_paths = load_images(data_dir)
        print(f"Loading dataset with {len(self.image_paths)} images...")

        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.transform = transform
        self.noise_transform = noise_transform
        self.num_patches_per_batch = num_patches_per_batch
        # Pre-compute patch indices instead of patches
        self.patch_indices = self._compute_patch_indices()
        print(
            f"Computed {batch_size} x {len(self.patch_indices) // batch_size} patch indices"
        )

    # This function computes the patch indices for the dataset and allow for on demand patch extraction
    def _compute_patch_indices(self):
        patch_indices = []
        for img_idx, image_path in enumerate(self.image_paths):
            image = decode_any_image(image_path)
            h, w = image.shape[1], image.shape[2]

            for i in range(0, h - self.patch_size + 1, self.stride):
                for j in range(0, w - self.patch_size + 1, self.stride):
                    patch_indices.append((img_idx, i, j))
        
        print(f"Computed {self.batch_size} x {len(patch_indices)//self.batch_size} patch indices")

        # Remove indices to align with batch size
        n_to_remove = len(patch_indices) % self.batch_size
        if n_to_remove > 0:
            print(f"Removing {n_to_remove} patches to align with batch size")
            patch_indices = patch_indices[:-n_to_remove]

        if len(patch_indices) > self.num_patches_per_batch*self.batch_size:
            print(f"Removing {len(patch_indices) - self.num_patches_per_batch*self.batch_size} patches to align with {self.num_patches_per_batch*self.batch_size} patches per batch")
            patch_indices = patch_indices[:self.num_patches_per_batch*self.batch_size]

        return patch_indices

    def __len__(self):
        return len(self.patch_indices)

    def __getitem__(self, idx):
        img_idx, i, j = self.patch_indices[idx]

        # Load image and extract patch on demand
        image = decode_any_image(self.image_paths[img_idx])
        patch = image[:, i : i + self.patch_size, j : j + self.patch_size]

        if self.transform:
            patch = self.transform()(patch)

        patch = patch.to(torch.float32) / 255.0

        if self.noise_transform:
            noisy_patch = self.noise_transform()(patch)
        else:
            noisy_patch = patch

        return noisy_patch, patch