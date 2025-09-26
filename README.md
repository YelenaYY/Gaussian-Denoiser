# Gaussian Denoiser

## Authors and OS
- Team: Rongfei (Eric) Jin & Yue (Yelena) Yu
- OS used: macOS, Linux (Arch)

## Time Travel Days
- Yes: We will use the travel days

## Environment Setup

Make sure you have `uv` and `make` installed 
- [`uv` installation guide](https://docs.astral.sh/uv/getting-started/installation/)

then run the following commands to download all python packages
```bash
uv sync
```

## Prepare the data

All except `urban100` datasets can be git cloned from branch [merge](https://github.com/YelenaYY/Gaussian-Denoiser/tree/merge)

the scripts will try downloading the `urban100` 
- in case it failed, manually download it [here](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u)

Make sure all the zip or tar.gz files are download  into `data/compressed` folder


The following command will extract training and testing images from `data/compressed' folder

```bash
make data
```

## Train DnCNN-S, color DnCNN-B, DnCNN-3

Run the following command to train the models

```bash
# Model S (Grayscale)
make train_s

# Model B (Grayscale)
make train_b

# Model CB (Color)
make train_cb

# Model 3 (Color)
make train_3
```

The following commands will generate comparison plots and statistics files in the `results` folder

```bash
# Model S
make test_s

# Model B
make test_b

# Model Color B
make test_cb

# Model 3
make train_3

# Summary file
make summary
```

## Iterative Changelogs
Please refer to [CHANGELOG.md](CHANGELOG.md)