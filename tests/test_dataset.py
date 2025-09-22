import unittest
from denoiser.dataset import PatchDataset, BicubicDownThenUp, RandomSigmaGaussianNoise
from torchvision.io import decode_image
from skimage.metrics import peak_signal_noise_ratio


class TestDataset(unittest.TestCase):
    def test_dataset(self):
        dataset = PatchDataset(
            "data/train/TRAIN400",
            patch_size=40,
            stride=6,
            batch_size=128,
        )
        self.assertEqual(len(dataset), 204800)
        noisy_patches, patches = dataset[0]
        self.assertEqual(tuple(noisy_patches.shape), (1, 40, 40))
        self.assertEqual(tuple(patches.shape), (1, 40, 40))

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
