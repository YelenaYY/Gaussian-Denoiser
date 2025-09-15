.PHONY: data train clean

# Training settings
MODEL_A_DIR = models/model_a
LOG_A_DIR = logs/model_a

MAX_EPOCH = 5
BATCH_SIZE = 128

# Testing settings
TEST_A_DIR = data/test/Set12 
MODEL_A_DIR = models/model_a
OUTPUT_A_DIR = results/model_a
NOISE_LEVEL_A = 25

# Data settings
DATA_DIR = data
TEST_DIR = $(DATA_DIR)/test
TRAIN_DIR = $(DATA_DIR)/train
COMPRESSED_DIR = $(DATA_DIR)/compressed


TRAIN400_DIR = $(TRAIN_DIR)/TRAIN400

BSDS200_ZIP = $(COMPRESSED_DIR)/BSDS200.zip
T91_ZIP = $(COMPRESSED_DIR)/T91.zip
TRAIN400_ZIP = $(COMPRESSED_DIR)/TRAIN400.zip
TEST_ZIP = $(COMPRESSED_DIR)/TEST.zip

data:
	mkdir -p $(TEST_DIR)
	mkdir -p $(TRAIN_DIR)

	unzip -o $(BSDS200_ZIP) -d $(TRAIN_DIR) 
	unzip -o $(T91_ZIP) -d $(TRAIN_DIR) 
	unzip -o $(TEST_ZIP) -d $(TEST_DIR) 

	mkdir -p $(TRAIN400_DIR)
	unzip -o $(TRAIN400_ZIP) -d $(TRAIN400_DIR)


train_a:
	mkdir -p $(MODEL_A_DIR)
	mkdir -p $(LOG_A_DIR)

	uv run main.py --data_dir $(TRAIN400_DIR) --model_dir $(MODEL_A_DIR) --log_dir $(LOG_A_DIR) --max_epoch $(MAX_EPOCH) --batch_size $(BATCH_SIZE)

test_a:
	mkdir -p $(OUTPUT_A_DIR)

	uv run main.py --mode test --test_dir $(TEST_A_DIR) --output_dir $(OUTPUT_A_DIR) --model_dir $(MODEL_A_DIR) --noise_level $(NOISE_LEVEL_A)

clean:
	rm -rf $(TEST_DIR)
	rm -rf $(TRAIN_DIR)