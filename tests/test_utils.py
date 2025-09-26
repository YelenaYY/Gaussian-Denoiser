# Unit tests for the dataset
# Not intended to be evaluated

import unittest
from pathlib import Path
from denoiser.utils import Logger

class TestLogger(unittest.TestCase):
    def test_logging_data(self):
        test_log_path = Path("unittest.log")
        logger = Logger(test_log_path, ["epoch", "loss", "accuracy"])
        
        test_data = {
            "epoch": 1,
            "loss": 0.1234,
            "accuracy": 0.95
        }
        logger.log(test_data)
        
        with open(test_log_path, 'r') as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 2)
        
        data_line = lines[1].strip()
        parts = data_line.split('\t')
        
        self.assertEqual(len(parts), 4)
        
        self.assertEqual(parts[1], "1")  # epoch
        self.assertEqual(parts[2], "0.1234")  # loss
        self.assertEqual(parts[3], "0.95")  # accuracy
        
        timestamp = parts[0]
        self.assertRegex(timestamp, r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}')

        # remove the log file
        test_log_path.unlink()


if __name__ == "__main__":
    unittest.main()
    