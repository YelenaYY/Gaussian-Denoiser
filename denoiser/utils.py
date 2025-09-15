from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp']

class Logger:
    def __init__(self, log_file_path: str | Path, headers: List[str]):
        if isinstance(log_file_path, str):
            log_file_path = Path(log_file_path)

        self.log_file_path = log_file_path
        self.headers = ['timestamp'] + headers
        self.file_initialized = False
        
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._initialize_log_file()
    
    def _initialize_log_file(self):
        if not self.log_file_path.exists():
            with open(self.log_file_path, 'w') as f:
                f.write('\t'.join(self.headers) + '\n')
        self.file_initialized = True
    
    def log(self, data: Dict[str, Any]):
        if not self.file_initialized:
            self._initialize_log_file()
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Include milliseconds
        
        row_data = [timestamp]
        
        for header in self.headers[1:]:  # Skip timestamp header
            value = data.get(header, '')
            row_data.append(str(value))
        
        with open(self.log_file_path, 'a') as f:
            f.write('\t'.join(row_data) + '\n')

def load_images(data_dir: Path):
    print(f"Loading images from {data_dir}")
    image_paths = []
    for extension in EXTENSIONS:
        image_paths.extend(list(Path(data_dir).glob(f"*{extension}")))
    return image_paths