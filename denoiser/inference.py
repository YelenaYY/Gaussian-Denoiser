from denoiser.model import DnCNN, load_checkpoint
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from torchvision.io import decode_image

from denoiser.utils import load_images

def inference(options: dict):
    test_dir = options['test_dir'] if 'test_dir' in options else 'data/Test'
    output_dir = options['output_dir'] if 'output_dir' in options else 'results'
    model_dir = options['model_dir'] if 'model_dir' in options else 'models'
    noise_level = options['noise_level'] if 'noise_level' in options else 25

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(model_dir)

    test_dir = Path(test_dir)
    image_paths = load_images(test_dir)

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {test_dir}")

    use_cuda = torch.cuda.is_available()

    model = DnCNN()
    if use_cuda:
        model.cuda()

    load_checkpoint(model, model_dir)

    for image_path in image_paths:
        process_image(model, image_path, use_cuda, noise_level, output_dir)

def process_image(model, image_path, use_cuda, noise_level, output_dir):
    image = decode_image(str(image_path))
    image = image.to(torch.float32) / 255.0

    noisy_image = image + torch.randn_like(image) * noise_level/255.0

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

    save_comparison_plot(original_image_np, noisy_image_np, denoised_image, output_dir / f'{image_path.stem}.png')


    # Plot comparison
def save_comparison_plot(original_image_np, noisy_image_np, denoised_image, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_image_np, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(noisy_image_np, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Noisy')
    axes[1].axis('off')

    axes[2].imshow(denoised_image, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Denoised')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()