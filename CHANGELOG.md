# Change log

The current file contains iterative changelogs. Since 2 branches existed before the final merge, the change log is split into 2 sections.

## Merge
- **Patch noise generation decision**: Direct use corresponding transformation provided `torchvision.transforms.v2`
- **Patch extraction decision**: if random images are selected for random patch extraction, we might miss certain detail-abundant patches due to the sampling process. Instead, we generate patches for each images and adjust the number by stride size

```python
# AFTER
def _compute_patch_indices(self):
        patch_indices = []
        for img_idx, image_path in enumerate(self.image_paths):
            image = decode_any_image(image_path)
            h, w = image.shape[1], image.shape[2]

            for i in range(0, h - self.patch_size + 1, self.stride):
                for j in range(0, w - self.patch_size + 1, self.stride):
                    patch_indices.append((img_idx, i, j))
# ...

# BEFORE
def _open_random_image(self) -> Image.Image:
  idx = random.randrange(len(self.files))
  return Image.open(self.files[idx]).convert("RGB")
```

- **Optimizer decision**: use Adam and torch-provided LR control for faster convergence compared to SGD

```python
# AFTER
# LR Exponentially decrease from 1e-4 at epoch 1 to 1e-1 at epoch 50
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = ExponentialLR(
    optimizer, gamma = (1e-4/1e-10)**(1/50) 
)


# BEFORE
for g in opt.param_groups:
  g['lr'] = exp_lr(ep, cfg.epochs, cfg.lr_start, cfg.lr_end)

# ...for each batch...
opt = torch.optim.SGD(model.parameters(), lr=cfg.lr_start, momentum=0.9, weight_decay=cfg.weight_decay)
```

## Branch A

#### September 21-23, 2025 - Model Development & Training Phase

**September 23, 2025**

- Minor updates and maintenance

**September 22, 2025**

- **Visualization Enhancement**: Added plotting functionality for model results
- **Model Training - DnCNN-B**:
  - Completed extended training with 50 epochs
  - Updated model results and performance metrics
- **Model Training - DnCNN-S**:
  - Completed training with 30 epochs
  - Generated training logs (PSNR) and results
- **Documentation**: Updated training logs across multiple training sessions

**September 21, 2025**

- **Major Update**: Implemented DnCNN architecture variants
  - Introduced DnCNN-S model
  - Introduced DnCNN-B model
  - Refactored training pipeline
  - Create dataloader for improved data handling
  - Fix Gradient Exploding Problem by clamping the loss

```python
# NEW ADDED
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### September 7, 2025 - Project Initialization

- Initial project setup and repository creation

## Branch B

#### September 21-23, 2025 - Testing, Documentation & Polish Phase

**September 23, 2025**

- **Documentation**: Added explanatory comments throughout codebase

**September 22-23, 2025**

- **Documentation**: Updated README.md with project details and usage instructions
- **Testing Enhancement**:
  - Expanded test suite with additional test sets
  - Implemented statistical output for performance analysis
- **Pipeline Refinement**: Updated data processing pipeline and dataset handling, particularly use on-demand patch extraction loading to reduce memory use

```python
# NEW
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
```

```python
# OLD
for i in range(0, image.shape[1] - self.patch_size + 1, self.stride):
    for j in range(0, image.shape[2] - self.patch_size + 1, self.stride):
        patch = image[:, i : i + self.patch_size, j : j + self.patch_size]

        if self.transform:
            patch = self.transform(patch)
        patches.append(patch)

n_patches_to_remove_for_batch_normalization = len(patches) % self.batch_size
patches = patches[:-n_patches_to_remove_for_batch_normalization]
patches = torch.stack(patches)
if self.normalize:
    patches = patches.to(torch.float32) / 255.0
return patches
```

**September 21, 2025**

- **Metrics Implementation**: Added comprehensive metrics output for model evaluation
- **Model Update**: Integrated DnCNN-S and DnCNN-B model variants (collaboration with Yelena)

#### September 15, 2025 - Architecture Refactoring

**September 15, 2025**

- **Data Management**: Uploaded and organized datasets
- **Project Structure**:
  - Converted Jupyter notebook prototype to structured Python package
  - Established data processing pipeline framework

#### September 14, 2025 - Model Development

**September 14, 2025**

- **Bug Fix**: Resolved forward calculation issues in model,

```python
# NEW version where the model directly output denoised image, 
# This change simplifies the training/inference process
def forward(self, x):
    y = x
    denoised = y - self.dncnn(x) # the model will output the noise
    return denoised

# Training 
output = model(noisy_patches)  # cleaned patches
target = patches
loss = criterion(output, target) 
epoch_loss += loss.item()

# OLD
def forward(self, x):
      return self.dncnn(x)

# OLD Training 
noise = model(noisy_patches)  # noise
target = patches
loss = criterion(output, target-noise) # Incorrect, we want to the model to learn the noise not final image
epoch_loss += loss.item()

```

- **Model Extension**: Implemented DnCNN-A variant with training capabilities
- **Patch Extraction**: Implemented Dataloader to extract patch, loading all patches into memory at once
- **Exploding Gradient Fix**

```python
# NEW, instead of SGD, Adam is used for more adaptive learning rate adjustment
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# OLD
optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4, momentum=0.9)
```

#### September 13, 2025 - Project Genesis

- **Initial Development**: Created initial scratch work and proof of concept
