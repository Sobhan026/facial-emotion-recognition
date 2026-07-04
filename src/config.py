from pathlib import Path


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
DEMO_DIR = OUTPUTS_DIR / "demo"

FER2013_CSV_PATH = RAW_DATA_DIR / "fer2013.csv"


# ---------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------

IMAGE_HEIGHT = 48
IMAGE_WIDTH = 48
IMAGE_CHANNELS = 1

IMAGE_SHAPE = (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_CHANNELS,
)

NUMBER_OF_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH

NUMBER_OF_CLASSES = 7

EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

EMOTION_NAMES = [
    EMOTION_LABELS[index]
    for index in range(NUMBER_OF_CLASSES)
]


# ---------------------------------------------------------
# Official FER2013 splits
# ---------------------------------------------------------

TRAIN_SPLIT_NAME = "Training"
VALIDATION_SPLIT_NAME = "PublicTest"
TEST_SPLIT_NAME = "PrivateTest"

VALID_USAGE_VALUES = {
    TRAIN_SPLIT_NAME,
    VALIDATION_SPLIT_NAME,
    TEST_SPLIT_NAME,
}


# ---------------------------------------------------------
# Preprocessing configuration
# ---------------------------------------------------------

NORMALIZATION_DIVISOR = 255.0

USE_HISTOGRAM_EQUALIZATION = False


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

RANDOM_SEED = 42


# ---------------------------------------------------------
# Training defaults
# ---------------------------------------------------------

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001


# ---------------------------------------------------------
# Inference configuration
# ---------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.60
REJECTION_LABEL = "Uncertain"


# ---------------------------------------------------------
# Directory creation
# ---------------------------------------------------------

def create_project_directories() -> None:
    """Create project output directories if they do not already exist."""

    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        DEMO_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Create required directories when this module is imported.
create_project_directories()