.PHONY: data train clean

MAX_EPOCH = 5
BATCH_SIZE = 128

# Testing settings
TEST_S_DIR = data/test/Set12
OUTPUT_S_DIR = results/model_s

# Data settings
DATA_DIR = data
TEST_DIR = $(DATA_DIR)/test
TRAIN_DIR = $(DATA_DIR)/train
COMPRESSED_DIR = $(DATA_DIR)/compressed


TRAIN400_DIR = $(TRAIN_DIR)/TRAIN400
CBSD68_DIR = $(TEST_DIR)/CBSD68

CBSD432_TAR = $(COMPRESSED_DIR)/CBSD432.tar.gz
BSDS200_ZIP = $(COMPRESSED_DIR)/BSDS200.zip
T91_ZIP = $(COMPRESSED_DIR)/T91.zip
TEST_ZIP = $(COMPRESSED_DIR)/TEST.zip

TRAIN400_ZIP = $(COMPRESSED_DIR)/TRAIN400.zip
CBSD68_ZIP = $(COMPRESSED_DIR)/CBSD68.zip

data:
	mkdir -p $(TEST_DIR)
	mkdir -p $(TRAIN_DIR)

	unzip -o $(BSDS200_ZIP) -d $(TRAIN_DIR) 
	unzip -o $(T91_ZIP) -d $(TRAIN_DIR) 
	unzip -o $(TEST_ZIP) -d $(TEST_DIR) 

	mkdir -p $(TRAIN400_DIR)
	unzip -o $(TRAIN400_ZIP) -d $(TRAIN400_DIR)
	unzip -o $(CBSD68_ZIP) -d $(CBSD68_DIR)

	tar -xzvf $(CBSD432_TAR) -C $(TRAIN_DIR)


train_s:
	uv run main.py --model_type s --train_data $(TRAIN400_DIR) --max_epoch $(MAX_EPOCH) --batch_size $(BATCH_SIZE)

test_s:
	uv run main.py --mode test --model_type s --test_data $(TEST_S_DIR)

train_cb:
	uv run main.py --model_type cb --train_data $(TRAIN_DIR)/CBSD432 --max_epoch $(MAX_EPOCH) --batch_size $(BATCH_SIZE)

test_cb:
	uv run main.py --mode test --model_type cb --test_data $(CBSD68_DIR)

clean:
	rm -rf $(TEST_DIR)
	rm -rf $(TRAIN_DIR)
