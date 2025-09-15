# Gaussian Denoiser

## Project Overview
TBA

## Environment Setup

Make sure you have `uv` installed

```bash
uv sync
```

## Prepare the data

The following command will extract training and testing images from `data/compressed' folder

```bash
make data
```

## Train and test DnCNN-A
Run the following command to train the model A
```bash
make train_a 
```
then, test the model with Set12 dataset

```bash
make test_a
```

