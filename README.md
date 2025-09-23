# Gaussian Denoiser

## Environment Setup

Make sure you have `uv` and `make` installed

```bash
uv sync
```

## Prepare the data

All except `urban100` datasets can be downloaded from [dev1](https://github.com/YelenaYY/Gaussian-Denoiser/tree/dev1)

to download urban100, go to [gdrive](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u) and download the zip to `data/compressed` folder



The following command will extract training and testing images from `data/compressed' folder

```bash
make data
```

## Train DnCNN-S, color DnCNN-B, DnCNN-3

Run the following command to train the model A

```bash
make train_s
make train_cb
make train_3
```

```bash
make test_s
make test_cb
make train_3
```

By default, an results folder will be created to store all the output