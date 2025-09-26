# Gaussian-Denoiser

## Authors and OS
- Team: Rongfei (Eric) Jin & Yue (Yelena) Yu
- OS used: macOS, Linux

## Time Travel Days
- Decision: No, We do not plan to use any time travel days.

## Setup (no compilation needed)
This project is Python-based. Use Conda to create the environment defined in `environment.yml`.

1. Create the environment (one-time):
   ```bash
   conda env create -f environment.yml
   ```
2. Activate it:
   ```bash
   conda activate gaussian-denoiser
   ```

## Dataset - Urban100
- If use Urban100, Please download from [Google Drive](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u) and place it under `data/compressed`.

## How to Run

### Training
- DnCNN-S (specific noise level σ):
  ```bash
  python -m dncnn.train --variant S --train_root data/Train400 \
    --sigma 25 --save "Training results/dncnn_S_sigma25_50epochs.pt"
  ```

- DnCNN-B (blind/general σ∈[0,55]):
  ```bash
  python -m dncnn.train --variant B --train_root data/Train400 \
    --save "Training results/dncnn_B_50epochs.pt"
  ```

Notes:
- Default depths follow the paper (S: 17, B: 20) and are set in the code.
- Metrics are printed each epoch and also appended to `metrics.csv` in the project root.

### Inference (Denoising)
- Single image (grayscale by default):
  ```bash
  python -m dncnn.inference --model "Training results/dncnn_B_50epochs.pt" \
    --input data/CBSD68-dataset/noisy25/0000.png --output denoised_0000.png
  ```

- Single image (color): add `--color`:
  ```bash
  python -m dncnn.inference --model "Training results/dncnn_B_50epochs.pt" \
    --input data/CBSD68-dataset/noisy25/0000.png --output denoised_0000.png --color
  ```

- Batch denoise a directory:
  ```bash
  python -m dncnn.inference --model "Training results/dncnn_B_50epochs.pt" \
    --input data/CBSD68-dataset/noisy25 --batch --output data/CBSD68-dataset/noisy25_denoised
  ```

### Evaluate SISR PSNR/SSIM (Y channel)
Evaluates average PSNR/SSIM on Set5 and Set14 as described in the DnCNN paper:
```bash
python evaluate_sisr_ssim.py \
  --model_s "Training results/dncnn_S_sigma25_50epochs.pt" \
  --model_b "Training results/dncnn_B_50epochs.pt"
```

## Note on multiple model versions
- We maintain two independent model versions to work in parallel on different aspects of the project （e.g., denoising‑centric modeling experimentation, model training and performance evaluation）
- Due to diverging design choices, differing training pipelines/dependencies, and time constraints near the deadline, we chose not to merge them to avoid destabilizing results.
- Please refer to each version's folder for its code structure and usage instructions.

## Repository Structure (key files)
- `dncnn/train.py`: training entry point and loops
- `dncnn/models.py`: DnCNN model
- `dncnn/datasets.py`: patch dataset and degradations (Gaussian, SISR, JPEG)
- `dncnn/utils.py`: utilities (PSNR, YCbCr conversion, file listing, image saving)
- `dncnn/inference.py`: single/batch denoising CLI and evaluation helpers
- `evaluate_sisr_ssim.py`: SISR evaluation script (PSNR/SSIM on Y channel)
- `environment.yml`: Conda environment specification

## Requirements
Created automatically by `environment.yml` (Python 3.12, PyTorch, torchvision).


