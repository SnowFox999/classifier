# config.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "BCN20000_prepared" 

METADATA_DIR = PROJECT_ROOT / "BCN20000" 

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100

# optimizer
LR = 1e-4

WEIGHT_DECAY = 1e-4

# training
EPOCHS = 100

PATIENCE = 20
NUM_WORKERS = 19

SEED = 42

TEST_SIZE = 0.15
VAL_SIZE = 0.15



