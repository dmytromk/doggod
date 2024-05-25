from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.as_posix()
SRC_DIR = BASE_DIR + "/src"
RESOURCES_DIR = BASE_DIR + "/resources"

EPOCHS = 10

BATCH_SIZE = 20
SHUFFLE_SEED = 158
IMG_SIZE = 224
