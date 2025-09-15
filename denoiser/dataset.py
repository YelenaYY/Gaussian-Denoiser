from torch.utils.data import Dataset
import torch
import random
from torchvision.io import decode_image
from torchvision import transforms
from torchvision.transforms import v2
from denoiser.utils import load_images


class PatchDataset(Dataset):
    def __init__(self, data_dir,patch_size, stride, batch_size, transform=None, scale_transform=None, normalize=True, noise_level:tuple[float, float]=(0,0)):
        self.image_paths = load_images(data_dir)

        print(f"Loading dataset with {len(self.image_paths)} images...")

        self.normalize = normalize
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.transform = transform
        self.scale_transform = scale_transform
        self.patches = self._read_all_patches() # read all patches
        self.noise_level = noise_level

        print(f"Loaded {batch_size}x{len(self.patches)//batch_size} patches")


    
    def __len__(self):
        return len(self.patches)

    def _extract_patches(self, idx):
        image_path = self.image_paths[idx]
        image = decode_image(str(image_path))

        if self.scale_transform:
            image = self.scale_transform(image)

        patches = []
        for i in range(0, image.shape[1] - self.patch_size + 1, self.stride):
            for j in range(0, image.shape[2] - self.patch_size + 1, self.stride):
                patch = image[:,i:i+self.patch_size, j:j+self.patch_size]
                if self.transform:
                    patch = self.transform(patch)
                patches.append(patch)
        
        n_patches_to_remove_for_batch_normalization = len(patches) % self.batch_size
        patches = patches[:-n_patches_to_remove_for_batch_normalization]
        patches = torch.stack(patches)
        if self.normalize:
            patches = patches.to(torch.float32) / 255.0
        return patches
    
    def _read_all_patches(self):
        all_patches = []
        for i in range(len(self.image_paths)):
            patches = self._extract_patches(i)
            all_patches.append(patches)
        return torch.cat(all_patches)
    
    def __getitem__(self, idx):
        patches = self.patches[idx]
        if isinstance(self.noise_level, tuple):
            noise_level = random.uniform(self.noise_level[0], self.noise_level[1])
            noisy_patches = patches + torch.randn_like(patches) * (noise_level / 255.0)
        return noisy_patches, patches

deafult_transform = transforms.RandomChoice([
    transforms.RandomRotation((-90, -90)),
    transforms.RandomRotation((-180, -180)),
    transforms.RandomRotation((-270, -270)),

    transforms.RandomVerticalFlip(1),
    transforms.Compose([
        transforms.RandomRotation((-90, -90)),
        transforms.RandomVerticalFlip(1),
    ]),

    transforms.Compose([
        transforms.RandomRotation((-180, -180)),
        transforms.RandomVerticalFlip(1),
    ]),

    transforms.Compose([
        transforms.RandomRotation((-270, -270)),
        transforms.RandomVerticalFlip(1),
    ]),
])

default_scale_transform = transforms.RandomChoice([
    # Down sampling then up sampling
    transforms.RandomChoice([
        v2.JPEG((5,99)),
    ]),
])
