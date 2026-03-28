import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Prepare data
BASE_DATA_PATH = os.path.join(_PROJECT_ROOT, 'data')
OUTPUT_CLS_PATH = os.path.join(_PROJECT_ROOT, 'data', 'data_cls')

# Training config
MOBILE_EPOCHS = 100
MOBILE_BATCH_SIZE = 32
MOBILE_IMG_SIZE = 224
MOBILE_DATA_DIR = OUTPUT_CLS_PATH
NUM_WORKERS = 2
