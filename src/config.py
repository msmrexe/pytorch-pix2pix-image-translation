"""
Configuration file for all project constants and hyperparameters.
"""
import torch

# System Configuration
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Configuration
DATASET_ROOT = "cityscapes_pix2pix_dataset"
TRAIN_DIR = f"{DATASET_ROOT}/train"
VAL_DIR = f"{DATASET_ROOT}/val"
IMG_SIZE = 256
CHANNELS_IMG = 3
# Normalization parameters for transforms
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

# Training Hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE = 16  # Reduced from 32 for potentially better results
LEARNING_RATE = 2e-4
BETA1 = 0.5
BETA2 = 0.999
LAMBDA_L1 = 100

# Logging and Output
LOG_FILE = "logs/pix2pix.log"
OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"
SAMPLE_DIR = f"{OUTPUT_DIR}/samples"
PLOT_DIR = f"{OUTPUT_DIR}/plots"
DISPLAY_INTERVAL = 10 # In epochs
