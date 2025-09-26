# Authors:Rongfei Jin and Yelena Yu,
# Date: 2025-09-23,
# Course: CS 7180 Advanced Perception

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from tqdm import tqdm
from pathlib import Path

from denoiser.model import DnCNN, load_latest_checkpoint
from denoiser.utils import decode_any_image, load_images
from denoiser.dataset import RandomSigmaGaussianNoise, BicubicDownThenUp, FloatJPEG


# The function is used to load the inference model.
def load_inference_model(options: dict):
    model_dir = options["model_dir"] if "model_dir" in options else "models"
    model_type = options["model_type"]

    if model_type == "s":
        image_channels = 1
        num_layers = 17

    elif model_type == "b":
        image_channels = 1
        num_layers = 20

    elif model_type == "cb":
        image_channels = 3
        num_layers = 20
    elif model_type == "3":
        image_channels = 3
        num_layers = 20
    else:
        raise ValueError("Invalid model type! must be one of s/b/3")

    model_dir = Path(model_dir)

    use_cuda = torch.cuda.is_available()

    model = DnCNN(num_layers=num_layers, image_channels=image_channels)
    if use_cuda:
        model.cuda()

    load_latest_checkpoint(model, model_dir / model_type)
    return model, use_cuda


# The function is used to create the noise generators based on the model type.
def create_noise_generators(model_type: str):
    noise_generators = []
    force_rgb = False

    if model_type == "s" or model_type == "b":
        noise_generators.append(("sigma15", RandomSigmaGaussianNoise((15 / 255.0, 15 / 255.0))))
        noise_generators.append(("sigma25", RandomSigmaGaussianNoise((25 / 255.0, 25 / 255.0))))
        noise_generators.append(("sigma50", RandomSigmaGaussianNoise((50 / 255.0, 50 / 255.0))))

    elif model_type == "cb":
        noise_generators.append(("sigma15", RandomSigmaGaussianNoise((15 / 255.0, 15 / 255.0))))
        noise_generators.append(("sigma25", RandomSigmaGaussianNoise((25 / 255.0, 25 / 255.0))))
        noise_generators.append(("sigma50", RandomSigmaGaussianNoise((50 / 255.0, 50 / 255.0))))
        force_rgb = True

    elif model_type == "3":
        noise_generators.append(("sigma15", RandomSigmaGaussianNoise((15 / 255.0, 15 / 255.0))))
        noise_generators.append(("sigma25", RandomSigmaGaussianNoise((25 / 255.0, 25 / 255.0))))
        noise_generators.append(("sigma50", RandomSigmaGaussianNoise((50 / 255.0, 50 / 255.0))))
        noise_generators.append(("bicubic2", BicubicDownThenUp([2])))
        noise_generators.append(("bicubic3", BicubicDownThenUp([3])))
        noise_generators.append(("bicubic4", BicubicDownThenUp([4])))
        noise_generators.append(("jpeg10", FloatJPEG((10, 10))))
        noise_generators.append(("jpeg20", FloatJPEG((20, 20))))
        noise_generators.append(("jpeg30", FloatJPEG((30, 30))))
        noise_generators.append(("jpeg40", FloatJPEG((40, 40))))
        force_rgb = True
    else:
        raise ValueError("Invalid model type! must be one of s/b/3")

    return noise_generators, force_rgb


# The function is used to process the noise generator on all images.
def process_noise_generator(
    model,
    use_cuda: bool,
    noise_generator_name: str,
    noise_generator,
    image_paths: list,
    test_output_dir: Path,
    force_rgb: bool,
):
    """Process a single noise generator on all images.
    
    Returns:
        pd.DataFrame: Statistics for this noise generator
    """
    # if the stats file already exists, skip
    if (test_output_dir / f"{noise_generator_name}_stats.csv").exists():
        return pd.read_csv(test_output_dir / f"{noise_generator_name}_stats.csv")
    
    stats = pd.DataFrame()
    for image_path in tqdm(image_paths, desc=f"Processing {noise_generator_name}"):
        noisy_psnr, denoised_psnr, noisy_ssim, denoised_ssim = process_image(
            model,
            image_path,
            use_cuda,
            noise_generator,
            test_output_dir / noise_generator_name,
            True,
            force_rgb,
        )

        stats = pd.concat(
            [
                stats,
                pd.DataFrame(
                    {
                        "image_path": [image_path],
                        "noisy_psnr": [noisy_psnr],
                        "noisy_ssim": [noisy_ssim],
                        "denoised_psnr": [denoised_psnr],
                        "denoised_ssim": [denoised_ssim],
                    }
                ),
            ],
            ignore_index=True,
        )

    stats.to_csv(test_output_dir / f"{noise_generator_name}_stats.csv", index=False)
    return stats


# The function is used to process the test dataset with all noise generators.
def process_test_dataset(
    model,
    use_cuda: bool,
    test_dir: str,
    model_type: str,
    output_dir: str,
    noise_generators: list[tuple[str, object]],
    force_rgb: bool,
):
    """Process a single test dataset with all noise generators."""
    image_paths = load_images(test_dir)
    if len(image_paths) == 0:
        print(f"No images found in {test_dir}")
        return

    test_set_name = Path(test_dir).stem
    test_output_dir = Path(output_dir) / model_type / test_set_name
    test_output_dir.mkdir(parents=True, exist_ok=True)

    average_stats = pd.DataFrame()

    for noise_generator_name, noise_generator in noise_generators:
        stats = process_noise_generator(
            model,
            use_cuda,
            noise_generator_name,
            noise_generator,
            image_paths,
            test_output_dir,
            force_rgb,
        )
        
        average_stats = pd.concat(
            [
                average_stats,
                pd.DataFrame(
                    {
                        "noise_type": [noise_generator_name],
                        "noisy_psnr": [stats["noisy_psnr"].mean()],
                        "noisy_ssim": [stats["noisy_ssim"].mean()],
                        "denoised_psnr": [stats["denoised_psnr"].mean()],
                        "denoised_ssim": [stats["denoised_ssim"].mean()],
                    }
                ),
            ],
            ignore_index=True,
        )

    average_stats.to_csv(test_output_dir / "average_stats.csv", index=False)

# The function is used to process the image and compute the metrics.
def process_image(
    model,
    image_path,
    use_cuda,
    noise_generator,
    output_dir,
    save_plot,
    force_rgb: bool = False,
):
    image = decode_any_image(image_path, force_rgb)
    image = image.to(torch.float32) / 255.0

    noisy_image = noise_generator(image)

    noisy_image = torch.clamp(noisy_image, 0, 1)

    noisy_image = noisy_image.unsqueeze(0)
    if use_cuda:
        noisy_image = noisy_image.cuda()

    model.eval()

    with torch.no_grad():
        denoised_image = model(noisy_image)
        denoised_image = torch.clamp(denoised_image, 0, 1)

    denoised_image = denoised_image.squeeze().detach().cpu().numpy()
    noisy_image_np = noisy_image.squeeze().detach().cpu().numpy()
    original_image_np = image.squeeze().numpy()

    if save_plot:
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        save_comparison_plot(
            original_image_np,
            noisy_image_np,
            denoised_image,
            output_dir / f"{image_path.stem}.png",
        )
    return compute_metrics(original_image_np, noisy_image_np, denoised_image)


# The function is used to compute the metrics between the original, noisy, and denoised images.
def compute_metrics(original, noisy, denoised) -> tuple[float, float, float, float]:
    noisy_psnr = peak_signal_noise_ratio(original, noisy, data_range=1)
    denoised_psnr = peak_signal_noise_ratio(original, denoised, data_range=1)
    if original.shape[0] == 3:
        color_channel = 0
    else:
        color_channel = None
    noisy_ssim = structural_similarity(original, noisy, data_range=1, channel_axis=color_channel)
    denoised_ssim = structural_similarity(original, denoised, data_range=1, channel_axis=color_channel)
    return noisy_psnr, denoised_psnr, noisy_ssim, denoised_ssim  # type: ignore


# The function is used to save the comparison plot between the original, noisy, and denoised images.
def save_comparison_plot(original_image_np, noisy_image_np, denoised_image, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    noisy_psnr, denoised_psnr, noisy_ssim, denoised_ssim = compute_metrics(original_image_np, noisy_image_np, denoised_image)

    if original_image_np.ndim == 2:
        original_image_np = np.expand_dims(original_image_np, axis=0)
    if noisy_image_np.ndim == 2:
        noisy_image_np = np.expand_dims(noisy_image_np, axis=0)
    if denoised_image.ndim == 2:
        denoised_image = np.expand_dims(denoised_image, axis=0)

    original_image_np = np.transpose(original_image_np, (1, 2, 0))
    noisy_image_np = np.transpose(noisy_image_np, (1, 2, 0))
    denoised_image = np.transpose(denoised_image, (1, 2, 0))

    axes[0].imshow(original_image_np, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(noisy_image_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Noisy (PSNR: {noisy_psnr:.2f}, SSIM: {noisy_ssim:.2f})")
    axes[1].axis("off")

    axes[2].imshow(denoised_image, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Denoised (PSNR: {denoised_psnr:.2f}, SSIM: {denoised_ssim:.2f})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# The function is used to test the model with all noise generators.
def test(options: dict):
    """Main test function that orchestrates the testing process."""
    model, use_cuda = load_inference_model(options)

    model_type = options["model_type"]
    test_data = options["test_data"]
    output_dir = options["output_dir"] if "output_dir" in options else "results"

    noise_generators, force_rgb = create_noise_generators(model_type)

    for test_dir in test_data:
        process_test_dataset(
            model,
            use_cuda,
            test_dir,
            model_type,
            output_dir,
            noise_generators,
            force_rgb,
        )

def extra(model_type: str):
    if model_type == "s":
        num_layers = 17
        image_channels = 1
        test_dir = Path("data") / "test" / "BSD68"
    elif model_type == "b":
        num_layers = 20
        image_channels = 1
        test_dir = Path("data") / "test" / "BSD68"
    elif model_type == "cb":
        num_layers = 20
        image_channels = 3
        test_dir = Path("data") / "test" / "CBSD68"
    elif model_type == "3":
        num_layers = 20
        image_channels = 3
        test_dir = Path("data") / "test" / "CBSD68"
    else:
        raise ValueError("Invalid model type! must be one of s/b/cb/3")
    
    additional_validation(model_type, num_layers, image_channels, test_dir)

def additional_validation(model_type: str, num_layers: int, image_channels: int, test_dir: str):
    # list model from epoch 1 to 50
    image_paths = load_images([test_dir])
    if len(image_paths) == 0:
        print(f"No images found in {test_dir}")
        return
    
    output_dir = Path("results") / "additional_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = pd.DataFrame()

    for epoch in range(50, 51):
        model_dir = Path("models") / model_type / f"model_{epoch:03d}.pth"
        if model_dir.exists():
            model = DnCNN(num_layers=num_layers, image_channels=image_channels)
            model.load_state_dict(torch.load(model_dir))
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()

            total_noisy_psnr = 0
            total_denoised_psnr = 0
            total_noisy_ssim = 0
            total_denoised_ssim = 0
            for image_path in tqdm(image_paths, desc=f"Processing epoch {epoch} for {model_dir}"):
                noisy_psnr, denoised_psnr, noisy_ssim, denoised_ssim = process_image(
                    model,
                    image_path,
                    torch.cuda.is_available(),
                    RandomSigmaGaussianNoise((25 / 255.0, 25 / 255.0)),
                    output_dir,
                    False,
                )
                print(noisy_psnr, denoised_psnr, noisy_ssim, denoised_ssim)
                total_noisy_psnr += noisy_psnr
                total_denoised_psnr += denoised_psnr
                total_noisy_ssim += noisy_ssim
                total_denoised_ssim += denoised_ssim
            stats = pd.concat(
                [
                    stats,
                    pd.DataFrame({"epoch": [epoch], "noisy_psnr": [total_noisy_psnr / len(image_paths)], "denoised_psnr": [total_denoised_psnr / len(image_paths)], "noisy_ssim": [total_noisy_ssim / len(image_paths)], "denoised_ssim": [total_denoised_ssim / len(image_paths)]}),
                ],
                ignore_index=True,
            )

    stats.to_csv(output_dir / f"{model_type}_stats.csv", index=False)
