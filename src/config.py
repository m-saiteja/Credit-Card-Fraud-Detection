from pathlib import Path
import logging

# PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# SCHEMA
TARGET_COL = 'Class'
TIME_COL = 'Time'
AMOUNT_COL = 'Amount'
PCA_COLS = [f'V{i}' for i in range(1, 29)]
FEATURE_COLS = PCA_COLS + [TIME_COL, AMOUNT_COL]
ALL_COLS = FEATURE_COLS + [TARGET_COL]

# SPLITS & REPRODUCABILITY
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.15
SPLIT_STRATEGY = "stratified"
VALID_SPLIT_STRATEGIES = ("stratified", "time")

#MLFlow
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT}/mlflow.db"
EXPERIMENT_NAME = "credit-card-fraud-detection"
REGISTERED_MODEL_NAME = "credit-card-fraud-classifier"
CHAMPION_ALIAS = "champion"


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S" 
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger