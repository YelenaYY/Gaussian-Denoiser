# Authors:Rongfei Jin and Yelena Yu,
# Date: 2025-09-23, 
# Course: CS 7180 Advanced Perception

from typing import Any
from torch.utils.data import Dataset
import torch
import random
from torchvision.transforms import v2
from denoiser.utils import decode_any_image, load_images


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


class RandomSigmaGaussianNoise:
    def __init__(self, noise: tuple[float, float] | float) -> None:
        if isinstance(noise, tuple):
            sigma = random.uniform(noise[0], noise[1])
            self.transform = v2.GaussianNoise(sigma=sigma)
        else:
            self.transform = v2.GaussianNoise(sigma=noise)
        pass

    def __call__(self, img: torch.Tensor):
        return self.transform(img)


class FloatJPEG:
    def __init__(self, quality: tuple[int, int]):
        self.quality = quality

    def __call__(self, img: torch.Tensor):
        return v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.JPEG(self.quality),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )(img)


MODEL_S_NOISE_TRANSFORM = RandomSigmaGaussianNoise(25.0 / 255.0)

MODEL_B_NOISE_TRANSFORM = RandomSigmaGaussianNoise((0, 55.0 / 255.0))

MODEL_3_NOISE_TRANSFORM = v2.RandomChoice(
    [
        FloatJPEG((5, 99)),
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
        transform=None,
        noise_transform: Any = MODEL_B_NOISE_TRANSFORM,
    ):
        self.image_paths = load_images(data_dir)
        print(f"Loading dataset with {len(self.image_paths)} images...")

        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.transform = transform
        self.noise_transform = noise_transform

        # Pre-compute patch indices instead of patches
        self.patch_indices = self._compute_patch_indices()
        print(
            f"Computed {batch_size} x {len(self.patch_indices) // batch_size} patch indices"
        )

    def _compute_patch_indices(self):
        patch_indices = []
        for img_idx, image_path in enumerate(self.image_paths):
            image = decode_any_image(image_path)
            h, w = image.shape[1], image.shape[2]

            for i in range(0, h - self.patch_size + 1, self.stride):
                for j in range(0, w - self.patch_size + 1, self.stride):
                    patch_indices.append((img_idx, i, j))

        # Remove indices to align with batch size
        n_to_remove = len(patch_indices) % self.batch_size
        if n_to_remove > 0:
            patch_indices = patch_indices[:-n_to_remove]

        return patch_indices

    def __len__(self):
        return len(self.patch_indices)

    def __getitem__(self, idx):
        img_idx, i, j = self.patch_indices[idx]

        # Load image and extract patch on demand
        image = decode_any_image(self.image_paths[img_idx])
        patch = image[:, i : i + self.patch_size, j : j + self.patch_size]

        if self.transform:
            patch = self.transform(patch)

        patch = patch.to(torch.float32) / 255.0

        if self.noise_transform:
            noisy_patch = self.noise_transform(patch)
        else:
            noisy_patch = patch

        return noisy_patch, patch


DEFAULT_TRANSFORM = v2.RandomChoice(
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

# Old version of dataset extraction, very slow
# New version is faster and more efficient by computing patches on demand

# def _extract_patches(self, idx):
#     image_path = self.image_paths[idx]

#     image = decode_image(str(image_path))

#     if self.scale_transform:
#         image = self.scale_transform(image)

#     patches = []

#     for i in range(0, image.shape[1] - self.patch_size + 1, self.stride):
#         for j in range(0, image.shape[2] - self.patch_size + 1, self.stride):
#             patch = image[:, i : i + self.patch_size, j : j + self.patch_size]

#             if self.transform:
#                 patch = self.transform(patch)

#             patches.append(patch)

#     n_patches_to_remove_for_batch_normalization = len(patches) % self.batch_size

#     patches = patches[:-n_patches_to_remove_for_batch_normalization]

#     patches = torch.stack(patches)

#     if self.normalize:
#         patches = patches.to(torch.float32) / 255.0

#     return patches


# def _read_all_patches(self):
#     all_patches = []

#     for i in range(len(self.image_paths)):
#         patches = self._extract_patches(i)

#         all_patches.append(patches)

#     return torch.cat(all_patches)


# def __getitem__(self, idx):
#     patches = self.patches[idx]

#     if isinstance(self.noise_level, tuple):
#         noise_level = random.uniform(self.noise_level[0], self.noise_level[1])

#         noisy_patches = patches + torch.randn_like(patches) * (noise_level / 255.0)

#     return noisy_patches, patches
