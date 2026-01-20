# config.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "BCN20000_prepared" 

METADATA_DIR = PROJECT_ROOT / "BCN20000" 

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100

# optimizer
LR_CLASSIFIER = 3e-5
LR_BACKBONE = 1e-4
LR_BACKBONE_FINETUNE = 3e-5
MIN_LR = 1e-6

WEIGHT_DECAY = 1e-4

# training
EPOCHS = 100
FINETUNE_EPOCH = 5   # с какой эпохи размораживаем всё

PATIENCE = 20
NUM_WORKERS = 19

SEED = 42

TEST_SIZE = 0.2
VAL_SIZE = 0.2



