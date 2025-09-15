import unittest
from denoiser.dataset import PatchDataset

class TestDataset(unittest.TestCase):
    def test_dataset(self):
        dataset = PatchDataset("data/train/TRAIN400", patch_size=40, stride=6, batch_size=128, noise_level=(0, 50))
        self.assertEqual(len(dataset), 204800)
        noisy_patches, patches = dataset[0]
        self.assertEqual(tuple(noisy_patches.shape), (1, 40, 40))
        self.assertEqual(tuple(patches.shape), (1, 40, 40))

if __name__ == "__main__":
    unittest.main()