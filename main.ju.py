# %%
import torch

# Check if cuda device is available
print("CUDA available:", torch.cuda.is_available())

# %%
# Load dataset
from torch.utils.data.dataset import Dataset  # noqa:E402
from torchvision.io import decode_jpeg  # noqa:E402
from pathlib import Path  # noqa:E402

# %%
#
#


class Bsd300(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = Path(data_folder)
        self.transform = transform
        self.labels = []
        with open(self.data_folder / "images/iids_train.txt") as f:
            self.labels = f.readlines()
        if len(self.labels) == 0:
            print("No labels found in", data_folder)

        pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img_name = self.data_folder / "/images/train/"
        image = decode_jpeg(img_name)
        if self.transform:
            image = self.transform(image)
        return image, label


# load from ./data/BSDS300/images/train/
