from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.as_posix()
SRC_DIR = BASE_DIR + "/src"
RESOURCES_DIR = BASE_DIR + "/resources"

BATCH_SIZE = 16
SHUFFLE_SEED = 158
IMG_SIZE = 224
