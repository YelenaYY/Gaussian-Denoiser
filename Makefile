# Authors:Rongfei Jin and Yelena Yu,
# Date: 2025-09-23, 
# Course: CS 7180 Advanced Perception
# File Description:
# This file contains the Makefile for the denoiser project.
# It includes the data preparation, training, testing, and summary generation.

.PHONY: data train clean

MAX_EPOCH = 50
BATCH_SIZE = 128
CHECKPOINT_DIR = models

# Testing settings

# Data settings
DATA_DIR = data
TEST_DIR = $(DATA_DIR)/test
TRAIN_DIR = $(DATA_DIR)/train
COMPRESSED_DIR = $(DATA_DIR)/compressed


# Dataset settings
TRAIN400_DIR = $(TRAIN_DIR)/TRAIN400
CBSD68_DIR = $(TEST_DIR)/CBSD68
SET12_DIR = $(TEST_DIR)/Set12
BSD68_DIR = $(TEST_DIR)/BSD68

SET14_DIR = $(TEST_DIR)/Set14
SET5_DIR = $(TEST_DIR)/Set5
CLASSIC5_DIR = $(TEST_DIR)/classic5
LIVE5_DIR = $(TEST_DIR)/LIVE1
URBAN100_DIR = $(TEST_DIR)/urban100
BSDS100_DIR = $(TEST_DIR)/BSDS100


CBSD432_TAR = $(COMPRESSED_DIR)/CBSD432.tar.gz
BSDS200_ZIP = $(COMPRESSED_DIR)/BSDS200.zip
T91_ZIP = $(COMPRESSED_DIR)/T91.zip
TEST_ZIP = $(COMPRESSED_DIR)/TEST.zip

TRAIN400_ZIP = $(COMPRESSED_DIR)/TRAIN400.zip
CBSD68_ZIP = $(COMPRESSED_DIR)/CBSD68.zip
BSDS100_ZIP = $(COMPRESSED_DIR)/BSDS100.zip
URBAN100_ZIP = $(COMPRESSED_DIR)/urban100.zip

data:
	bash check_data.sh

	mkdir -p $(TEST_DIR)
	mkdir -p $(TRAIN_DIR)

	unzip -o $(BSDS200_ZIP) -d $(TRAIN_DIR) 
	unzip -o $(T91_ZIP) -d $(TRAIN_DIR) 
	unzip -o $(TEST_ZIP) -d $(TEST_DIR) 

	mkdir -p $(TRAIN400_DIR)
	unzip -o $(TRAIN400_ZIP) -d $(TRAIN400_DIR)
	unzip -o $(CBSD68_ZIP) -d $(CBSD68_DIR)
	unzip -o $(BSDS100_ZIP) -d $(TEST_DIR)
	unzip -o $(URBAN100_ZIP) -d $(TEST_DIR)

	tar -xzvf $(CBSD432_TAR) -C $(TRAIN_DIR)


train_s:
	uv run main.py --model_type s --train_data $(TRAIN400_DIR) --max_epoch $(MAX_EPOCH) --batch_size $(BATCH_SIZE)

test_s:
	uv run main.py --mode test --model_type s --test_data $(SET12_DIR) $(BSD68_DIR)

train_b:
	uv run main.py --model_type b --train_data $(TRAIN400_DIR) --max_epoch $(MAX_EPOCH) --batch_size $(BATCH_SIZE)

test_b:
	uv run main.py --mode test --model_type b --test_data $(SET12_DIR) $(BSD68_DIR)

train_cb:
	uv run main.py --model_type cb --train_data $(TRAIN_DIR)/CBSD432 --max_epoch $(MAX_EPOCH) --batch_size $(BATCH_SIZE)

test_cb:
	uv run main.py --mode test --model_type cb --test_data $(CBSD68_DIR)

train_3:
	uv run main.py --model_type 3 --train_data $(TRAIN_DIR)/T91 $(TRAIN_DIR)/BSDS200 --max_epoch $(MAX_EPOCH) --batch_size $(BATCH_SIZE) --checkpoint $(CHECKPOINT_DIR)/cb/model_050.pth

train_3_resume:
	uv run main.py --model_type 3 --train_data $(TRAIN_DIR)/T91 $(TRAIN_DIR)/BSDS200 --max_epoch $(MAX_EPOCH) --batch_size $(BATCH_SIZE)

test_3:
	uv run main.py --mode test --model_type 3 --test_data $(CBSD68_DIR) $(SET14_DIR) $(SET5_DIR) $(CLASSIC5_DIR) $(LIVE5_DIR) $(BSDS100_DIR) $(URBAN100_DIR)

summary:
	uv run results.py

extra_s:
	uv run main.py --mode additional_validation --model_type s 
extra_b:
	uv run main.py --mode additional_validation --model_type b 
extra_cb:
	uv run main.py --mode additional_validation --model_type cb 
extra_3:
	uv run main.py --mode additional_validation --model_type 3 

clean:
	rm -rf $(TEST_DIR)
	rm -rf $(TRAIN_DIR)
