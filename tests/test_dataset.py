import unittest
from denoiser.dataset import PatchDataset, BicubicDownThenUp, RandomSigmaGaussianNoise, MODEL_3_NOISE_TRANSFORM
from torchvision.io import decode_image
from skimage.metrics import peak_signal_noise_ratio
from torchvision.transforms import v2
import torch


class TestDataset(unittest.TestCase):
    def test_dataset(self):
        dataset = PatchDataset(
            ["data/train/T91", "data/train/BSDS200"],
            patch_size=50,
            stride=7,
            batch_size=128,
            noise_transform=MODEL_3_NOISE_TRANSFORM,
        )
        print(len(dataset))
        for i in range(len(dataset)):
            noisy_patches, patches = dataset[i]
            self.assertEqual(tuple(noisy_patches.shape), (3, 50, 50))
            self.assertEqual(tuple(patches.shape), (3, 50, 50))


    def test_transform(self):
        img = decode_image("data/train/T91/t1.png")
        img = img.float() / 255
        bicubic = BicubicDownThenUp()
        transformed = bicubic(img)
        assert img.shape == transformed.shape

        psnr = peak_signal_noise_ratio(transformed.numpy(), img.numpy())
        assert 20 < psnr < 60, f"PSNR: {psnr}"

        sigma0 = RandomSigmaGaussianNoise(0)
        transformed = sigma0(img)
        psnr = peak_signal_noise_ratio(transformed.numpy(), img.numpy())
        print("sigma0 psnr", psnr)
        assert psnr > 50, f"PSNR: {psnr}"

        sigma_high = RandomSigmaGaussianNoise((0.1, 0.2))
        transformed = sigma_high(img)
        psnr = peak_signal_noise_ratio(transformed.numpy(), img.numpy())
        print("sigma_high psnr", psnr)
        assert psnr < 25, f"PSNR: {psnr}"
    
    def test_jpeg_noise(self):
        img = decode_image("data/train/T91/t1.png")
        img = img.float() / 255
        transformed = v2.Compose([v2.ToDtype(torch.uint8, scale=True), v2.JPEG((10, 10)), v2.ToDtype(torch.float32, scale=True)])(img)
        psnr = peak_signal_noise_ratio(transformed.numpy(), img.numpy())
        print("jpeg psnr", psnr)
        # comparison_plot(img, transformed)
        assert 20 < psnr < 30, f"PSNR: {psnr}"
    
def comparison_plot(img, transformed):
    img = img.numpy().transpose(1, 2, 0)
    transformed = transformed.numpy().transpose(1, 2, 0)
    from matplotlib import pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(transformed, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Transformed")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("comparison.png")
    plt.close()



if __name__ == "__main__":
    unittest.main(verbosity=2)
