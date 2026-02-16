# config.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "BCN20000/" 

METADATA_DIR = PROJECT_ROOT

SPLIT_FILE = PROJECT_ROOT / "experiment/BCN20000/master_split_file_new_data.csv"
#20 val, 60 train, => 20 test
#master_split_file_new_data_5_val.csv

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100
N_FOLDS = 5

LR = 1e-4
WEIGHT_DECAY = 1e-3
DROPOUT = 0.4

UNFREEZE = 10

PATIENCE = 20
NUM_WORKERS = 19

SEED = 42

TEST_SIZE = 0.20
VAL_SIZE = 0.20



