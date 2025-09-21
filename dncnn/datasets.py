# datasets.py - Dataset & degradations (Gaussian, SISR, JPEG)
import math
import random
import io
from typing import Tuple, List, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, to_pil_image


# ---------- Degradations (Gaussian, SISR, JPEG) ----------

def add_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    std = (sigma / 255.0)
    noise = torch.randn_like(x) * std
    return x + noise


def degrade_sisr(x: torch.Tensor, scale: int) -> torch.Tensor:
    b, c, h, w = x.shape
    down_h, down_w = max(1, h // scale), max(1, w // scale)
    imgs = [to_pil_image(x[i]) for i in range(b)]
    y = []
    for img in imgs:
        lr = img.resize((down_w, down_h), Image.BICUBIC)
        up = lr.resize((w, h), Image.BICUBIC)
        y.append(to_tensor(up))
    return torch.stack(y, dim=0).to(x.device)


def degrade_jpeg(x: torch.Tensor, quality: int) -> torch.Tensor:
    b = x.size(0)
    out = []
    for i in range(b):
        img = to_pil_image(x[i].cpu())
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        comp = Image.open(buf).convert("RGB" if x.size(1) == 3 else "L")
        out.append(to_tensor(comp))
    y = torch.stack(out, dim=0).to(x.device)

    # If grayscale:
    if x.size(1) == 1 and y.size(1) == 3:
        y = y.mean(dim=1, keepdim=True)
    return y


# ---------- Datasets ----------

class PatchDataset(Dataset):
    def __init__(
        self,
        filepaths: List[str],
        patch_size: int = 50,
        grayscale: bool = True,
        task: str = "gauss",                   # 'gauss' | 'multi'
        sigma_range: Tuple[float, float] = (0, 55),
        sisr_scales: Tuple[int, ...] = (2, 3, 4),
        jpeg_q_range: Tuple[int, int] = (5, 99),
        fixed_sigma: Optional[float] = None,   # set for DnCNN-S
        total_patches: int = 128 * 3000        # default to DnCNN-B scale
    ):
        self.files = filepaths
        if len(self.files) == 0:
            raise ValueError("No training images found.")
        self.ps = patch_size
        self.gray = grayscale
        self.task = task
        self.sigma_range = sigma_range
        self.scales = sisr_scales
        self.q_range = jpeg_q_range
        self.fixed_sigma = fixed_sigma
        self.total_patches = int(total_patches)

        tfms = [transforms.RandomCrop(self.ps),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip()]
        if self.gray:
            tfms = [transforms.Grayscale()] + tfms
        tfms += [transforms.ToTensor()]
        self.tf = transforms.Compose(tfms)

    def __len__(self):
        # Fixed-length "epoch" independent of dataset size
        return self.total_patches

    def _open_random_image(self) -> Image.Image:
        # Randomly choose a source image every sample
        idx = random.randrange(len(self.files))
        # Note: Convert to RGB; grayscale is handled by transform above
        return Image.open(self.files[idx]).convert("RGB")

    def __getitem__(self, idx):
        img = self._open_random_image()
        x = self.tf(img).clamp(0, 1)  # clean in [0,1], C=1 or 3

        if self.task == "gauss":
            sigma = self.fixed_sigma if self.fixed_sigma is not None else random.uniform(*self.sigma_range)
            y = add_gaussian_noise(x.unsqueeze(0), sigma).squeeze(0)

        elif self.task == "multi":
            choice = random.choice(["gauss", "sisr", "jpeg"])
            if choice == "gauss": # gauss
                sigma = random.uniform(*self.sigma_range)
                y = add_gaussian_noise(x.unsqueeze(0), sigma).squeeze(0)
            elif choice == "sisr": # sisr
                scale = random.choice(self.scales)
                y = degrade_sisr(x.unsqueeze(0), scale).squeeze(0)
            else:  # jpeg
                q = random.randint(*self.q_range)
                y = degrade_jpeg(x.unsqueeze(0), q).squeeze(0)
        
        else:
            raise ValueError(f"Unknown task: {self.task}")

        v = y - x
        return y, x, v