"""
Author: Yue (Yelena) Yu,  Rongfei (Eric) JIn
Date: 2025-09-23
Class: CS 7180 Advanced Perception
"""
# inference.py - Load model and denoise an image
import argparse
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from models import DnCNN
from utils import save_image, psnr, calculate_psnr_ycbcr


@torch.inference_mode()
def denoise_image(model_path: str, img_path: str, output_path: str = None, 
                  grayscale: bool = True, depth: int = 20) -> Image.Image:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_ch = 1 if grayscale else 3
    
    # Load model
    model = DnCNN(depth=depth, in_channels=in_ch)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    
    # Load and preprocess image
    img = Image.open(img_path).convert("L" if grayscale else "RGB")
    x = to_tensor(img).unsqueeze(0).to(device)
    
    # Denoise
    x_hat, _ = model(x)
    
    # Convert back to PIL Image
    result = to_pil_image(x_hat.squeeze(0).cpu().clamp(0, 1))
    
    # Save if output path is provided
    if output_path:
        result.save(output_path)
        print(f"Denoised image saved to: {output_path}")
    
    return result


@torch.inference_mode()
def evaluate_denoising(model_path: str, clean_img_path: str, noisy_img_path: str,
                       grayscale: bool = True, depth: int = 20) -> dict:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_ch = 1 if grayscale else 3
    
    # Load model
    model = DnCNN(depth=depth, in_channels=in_ch)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    
    # Load images
    clean_img = Image.open(clean_img_path).convert("L" if grayscale else "RGB")
    noisy_img = Image.open(noisy_img_path).convert("L" if grayscale else "RGB")
    
    clean_tensor = to_tensor(clean_img).unsqueeze(0).to(device)
    noisy_tensor = to_tensor(noisy_img).unsqueeze(0).to(device)
    
    # Denoise
    denoised_tensor, _ = model(noisy_tensor)
    
    # Calculate metrics
    clean_tensor = clean_tensor.squeeze(0)
    noisy_tensor = noisy_tensor.squeeze(0)
    denoised_tensor = denoised_tensor.squeeze(0)
    
    if grayscale:
        input_psnr = psnr(clean_tensor, noisy_tensor)
        output_psnr = psnr(clean_tensor, denoised_tensor)
    else:
        input_psnr = calculate_psnr_ycbcr(clean_tensor, noisy_tensor)
        output_psnr = calculate_psnr_ycbcr(clean_tensor, denoised_tensor)
    
    return {
        'input_psnr': input_psnr,
        'output_psnr': output_psnr,
        'improvement': output_psnr - input_psnr
    }


@torch.inference_mode()
def batch_denoise(model_path: str, input_dir: str, output_dir: str,
                  grayscale: bool = True, depth: int = 20):
    import os
    from utils import make_filelist
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_ch = 1 if grayscale else 3
    
    # Load model
    model = DnCNN(depth=depth, in_channels=in_ch)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    
    # Get all image files
    image_files = make_filelist(input_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(image_files)} images...")
    
    for i, img_path in enumerate(image_files):
        # Load and preprocess image
        img = Image.open(img_path).convert("L" if grayscale else "RGB")
        x = to_tensor(img).unsqueeze(0).to(device)
        
        # Denoise
        x_hat, _ = model(x)
        
        # Save result
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_denoised{ext}")
        
        result = to_pil_image(x_hat.squeeze(0).cpu().clamp(0, 1))
        result.save(output_path)
        
        print(f"Processed {i+1}/{len(image_files)}: {filename}")
    
    print(f"All images processed and saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DnCNN Inference")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--input", required=True, help="Input image or directory")
    parser.add_argument("--output", help="Output image or directory")
    parser.add_argument("--batch", action="store_true", help="Process directory of images")
    parser.add_argument("--depth", type=int, default=20, help="Model depth")
    parser.add_argument("--color", action="store_true", help="Process color images (default: grayscale)")
    parser.add_argument("--evaluate", help="Path to clean reference image for evaluation")
    
    args = parser.parse_args()
    
    grayscale = not args.color
    
    if args.evaluate:
        # Evaluation mode
        metrics = evaluate_denoising(args.model, args.evaluate, args.input, 
                                   grayscale=grayscale, depth=args.depth)
        print(f"Input PSNR: {metrics['input_psnr']:.2f} dB")
        print(f"Output PSNR: {metrics['output_psnr']:.2f} dB")
        print(f"Improvement: {metrics['improvement']:.2f} dB")
    elif args.batch:
        # Batch processing mode
        if not args.output:
            args.output = args.input + "_denoised"
        batch_denoise(args.model, args.input, args.output, 
                     grayscale=grayscale, depth=args.depth)
    else:
        # Single image mode
        result = denoise_image(args.model, args.input, args.output, 
                              grayscale=grayscale, depth=args.depth)