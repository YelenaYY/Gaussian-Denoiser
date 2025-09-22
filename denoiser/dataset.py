from torch.utils.data import Dataset
import torch
import random
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import v2
from denoiser.utils import decode_any_image, load_images
from multiprocessing import Pool


class BicubicDownThenUp:
    def __init__(self, scale_factors=[2, 3, 4]):
        self.scale_factors = scale_factors

    def __call__(self, img: torch.Tensor):
        original_size = img.shape[-2:]
        scale_factor = random.choice(self.scale_factors)

        down_h = max(1, original_size[0] // scale_factor)
        down_w = max(1, original_size[1] // scale_factor)

        print(down_h)

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

        return upsampled.to(torch.uint8)


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


MODEL_S_NOISE_TRANSFORM = RandomSigmaGaussianNoise(25.0 / 255.0)

MODEL_B_NOISE_TRANSFORM = RandomSigmaGaussianNoise((0, 55.0 / 255.0))

MODEL_3_NOISE_TRANSFORM = transforms.RandomChoice(
    [
        v2.JPEG((5, 99)),
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
        noise_transform=MODEL_B_NOISE_TRANSFORM,
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
        print(f"Computed {len(self.patch_indices)} patch indices")

    def _compute_patch_indices(self):
        patch_indices = []
        for img_idx, image_path in enumerate(self.image_paths):
            # Get image dimensions without loading full image
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


# class PatchDataset(Dataset):
#     def __init__(
#         self,
#         data_dir: str | list[str],
#         patch_size,
#         stride,
#         batch_size,
#         transform=None,
#         noise_transform=MODEL_B_NOISE_TRANSFORM,
#     ):
#         self.image_paths = load_images(data_dir)
#
#         print(f"Loading dataset with {len(self.image_paths)} images...")
#
#         self.patch_size = patch_size
#         self.stride = stride
#         self.batch_size = batch_size
#         self.transform = transform
#         self.noise_transform = noise_transform
#         # self.patches = self._read_all_patches()  # read all patches
#         self.patches = self._read_all_patches_parallel()
#
#         print(f"Loaded {batch_size}x{len(self.patches) // batch_size} patches")
#
#     def __len__(self):
#         return len(self.patches)
#
#     def _extract_patches(self, idx):
#         image_path = self.image_paths[idx]
#         image = decode_any_image(image_path)
#
#         patches = []
#         for i in range(0, image.shape[1] - self.patch_size + 1, self.stride):
#             for j in range(0, image.shape[2] - self.patch_size + 1, self.stride):
#                 patch = image[:, i : i + self.patch_size, j : j + self.patch_size]
#                 if self.transform:
#                     patch = self.transform(patch)
#                 patches.append(patch)
#
#         n_patches_to_remove_for_batch_normalization = len(patches) % self.batch_size
#         patches = patches[:-n_patches_to_remove_for_batch_normalization]
#         patches = torch.stack(patches)
#         patches = patches.to(torch.float32) / 255.0  # Convert to float [0,1]
#         return patches
#
#     def _read_all_patches(self):
#         all_patches = []
#         for i in tqdm(range(len(self.image_paths))):
#             patches = self._extract_patches(i)
#             all_patches.append(patches)
#         return torch.cat(all_patches)
#
#     def __getitem__(self, idx):
#         patches = self.patches[idx]  # [0,1]
#         # if isinstance(self.noise_level, tuple):
#         #     noise_level = random.uniform(self.noise_level[0], self.noise_level[1])
#         #     noisy_patches = patches + torch.randn_like(patches) * (noise_level / 255.0)
#         if self.noise_transform:
#             noisy_patches = self.noise_transform(patches)
#         return noisy_patches, patches
#
#     def _extract_patches_for_multiprocessing(self, idx):
#         # Static method for multiprocessing
#         image = decode_any_image(self.image_paths[idx])
#         patches = image.unfold(1, self.patch_size, self.stride).unfold(
#             2, self.patch_size, self.stride
#         )
#         patches = patches.contiguous().view(
#             -1, image.shape[0], self.patch_size, self.patch_size
#         )
#
#         n_to_remove = patches.shape[0] % self.batch_size
#         if n_to_remove > 0:
#             patches = patches[:-n_to_remove]
#
#         return patches.to(torch.float32) / 255.0
#
#     def _read_all_patches_parallel(self):
#         with Pool() as pool:
#             all_patches = list(
#                 tqdm(
#                     pool.imap(
#                         self._extract_patches_for_multiprocessing,
#                         range(len(self.image_paths)),
#                     ),
#                     total=len(self.image_paths),
#                     desc="Loading patches",
#                 )
#             )
#         return torch.cat(all_patches)
#

deafult_transform = transforms.RandomChoice(
    [
        transforms.RandomRotation((-90, -90)),
        transforms.RandomRotation((-180, -180)),
        transforms.RandomRotation((-270, -270)),
        transforms.RandomVerticalFlip(1),
        transforms.Compose(
            [
                transforms.RandomRotation((-90, -90)),
                transforms.RandomVerticalFlip(1),
            ]
        ),
        transforms.Compose(
            [
                transforms.RandomRotation((-180, -180)),
                transforms.RandomVerticalFlip(1),
            ]
        ),
        transforms.Compose(
            [
                transforms.RandomRotation((-270, -270)),
                transforms.RandomVerticalFlip(1),
            ]
        ),
    ]
)
