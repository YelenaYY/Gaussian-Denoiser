# Authors:Rongfei Jin and Yelena Yu,
# Date: 2025-09-23, 
# Course: CS 7180 Advanced Perception
# File Description:
# This file contains the utilities for the denoiser package.
# It includes the logger class, which is used to log the training process.
# It also includes the image loading function.

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import PIL.Image as Image

from torchvision.transforms import v2


# Define the extensions of the images to load.
EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"]


# This class is used to log the training process.
class Logger:
    def __init__(self, log_file_path: str | Path, headers: List[str]):
        if isinstance(log_file_path, str):
            log_file_path = Path(log_file_path)

        self.log_file_path = log_file_path
        self.headers = ["timestamp"] + headers
        self.file_initialized = False

        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        self._initialize_log_file()

    # This function is used to initialize the log file.
    def _initialize_log_file(self):
        if not self.log_file_path.exists():
            with open(self.log_file_path, "w") as f:
                f.write("\t".join(self.headers) + "\n")
        self.file_initialized = True

    # This function is used to log the data to the log file.
    def log(self, data: Dict[str, Any]):
        if not self.file_initialized:
            self._initialize_log_file()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[
            :-3
        ] 

        row_data = [timestamp]

        for header in self.headers[1:]:  # Skip timestamp header
            value = data.get(header, "")
            row_data.append(str(value))

        with open(self.log_file_path, "a") as f:
            f.write("\t".join(row_data) + "\n")


# This function is used to load the images from the data directories.
def load_images(data_dirs: str | list[str]) -> list[Path]:
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    image_paths = []
    for dir in data_dirs:
        print(f"Loading images from {dir}")
        for extension in EXTENSIONS:
            image_paths.extend(list(Path(dir).glob(f"*{extension}")))

    return image_paths


TO_TENSOR = v2.ToImage()


# This function is used to decode any image to a tensor use PIL.
def decode_any_image(image_path: Path, force_rgb: bool = False):
    image = Image.open(image_path)
    if force_rgb:
        image = image.convert("RGB")
    return TO_TENSOR(image)
