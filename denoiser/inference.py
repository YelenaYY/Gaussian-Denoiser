from denoiser.model import DnCNN, load_checkpoint
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from torchvision.io import decode_image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from denoiser.utils import decode_any_image, load_images
import numpy as np


def inference(options: dict):
    test_data = options["test_data"] if "test_data" in options else ["data/test/Set12"]
    output_dir = options["output_dir"] if "output_dir" in options else "results"
    model_dir = options["model_dir"] if "model_dir" in options else "models"
    noise_level = options["noise_level"] if "noise_level" in options else 25
    model_type = options["model_type"]

    if model_type == "s":
        image_channels = 1
        num_layers = 17

    elif model_type == "cb":
        image_channels = 3
        num_layers = 20
    else:
        raise ValueError("Invalid model type! must be one of s/b/3")

    output_dir = Path(output_dir) / model_type
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(model_dir)

    image_paths = load_images(test_data)

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {test_data}")

    use_cuda = torch.cuda.is_available()

    model = DnCNN(num_layers=num_layers, image_channels=image_channels)
    if use_cuda:
        model.cuda()

    load_checkpoint(model, model_dir / model_type)

    total_noisy_psnr, total_denoised_psnr, total_noisy_ssim, total_denoised_ssim = (
        0,
        0,
        0,
        0,
    )

    for image_path in image_paths:
        noisy_psnr, denoised_psnr, noisy_ssim, denoised_ssim = process_image(
            model, image_path, use_cuda, noise_level, output_dir, True
        )
        total_noisy_psnr += noisy_psnr
        total_denoised_psnr += denoised_psnr
        total_noisy_ssim += noisy_ssim
        total_denoised_ssim += denoised_ssim

    total_noisy_psnr /= len(image_paths)
    total_denoised_psnr /= len(image_paths)
    total_noisy_ssim /= len(image_paths)
    total_denoised_ssim /= len(image_paths)

    print(f"Total Noisy PSNR: {total_noisy_psnr:.2f}")
    print(f"Total Denoised PSNR: {total_denoised_psnr:.2f}")
    print(f"Total Noisy SSIM: {total_noisy_ssim:.2f}")
    print(f"Total Denoised SSIM: {total_denoised_ssim:.2f}")


def process_image(model, image_path, use_cuda, noise_level, output_dir, save_plot):
    image = decode_any_image(image_path)
    image = image.to(torch.float32) / 255.0

    noisy_image = image + torch.randn_like(image) * noise_level / 255.0

    # Clamp to valid range
    noisy_image = torch.clamp(noisy_image, 0, 1)

    # Prepare for model inference
    noisy_image = noisy_image.unsqueeze(0)
    if use_cuda:
        noisy_image = noisy_image.cuda()

    # Set model to evaluation mode
    model.eval()

    # Denoise the image
    with torch.no_grad():  # Disable gradients for inference
        denoised_image = model(noisy_image)
        # Clamp the result to valid range
        denoised_image = torch.clamp(denoised_image, 0, 1)

    # Convert back to numpy for visualization
    denoised_image = denoised_image.squeeze().detach().cpu().numpy()
    noisy_image_np = noisy_image.squeeze().detach().cpu().numpy()
    original_image_np = image.squeeze().numpy()

    if save_plot:
        save_comparison_plot(
            original_image_np,
            noisy_image_np,
            denoised_image,
            output_dir / f"{image_path.stem}.png",
        )
    return compute_metrics(original_image_np, noisy_image_np, denoised_image)


def compute_metrics(original, noisy, denoised) -> tuple[float, ...]:
    noisy_psnr = peak_signal_noise_ratio(original, noisy, data_range=1)
    denoised_psnr = peak_signal_noise_ratio(original, denoised, data_range=1)
    if original.shape[0] == 3:
        color_channel = 0
    else:
        color_channel = None
    noisy_ssim = structural_similarity(
        original, noisy, data_range=1, channel_axis=color_channel
    )
    denoised_ssim = structural_similarity(
        original, denoised, data_range=1, channel_axis=color_channel
    )
    return noisy_psnr, denoised_psnr, noisy_ssim, denoised_ssim  # type: ignore
    # Plot comparison


def save_comparison_plot(
    original_image_np, noisy_image_np, denoised_image, output_path
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    noisy_psnr, denoised_psnr, noisy_ssim, denoised_ssim = compute_metrics(
        original_image_np, noisy_image_np, denoised_image
    )

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
    axes[2].set_title(
        f"Denoised (PSNR: {denoised_psnr:.2f}, SSIM: {denoised_ssim:.2f})"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
