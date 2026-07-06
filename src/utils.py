import os
import logging
import sys

def setup_logger(name: str = "churn_prediction", log_level: int = logging.INFO) -> logging.Logger:
    """Sets up a standardized logger for the project."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(log_level)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler("logs/app.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def ensure_directories(dirs: list = None) -> None:
    """Ensures that required directories exist."""
    if dirs is None:
        dirs = ["data/raw", "data/processed", "models", "logs", "notebooks", "dashboard", "api"]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            print(f"Created directory: {d}")

class ChurnException(Exception):
    """Base exception class for Churn Prediction System."""
    pass

class DataPreprocessingError(ChurnException):
    """Raised when an error occurs during data preprocessing."""
    pass

class FeatureEngineeringError(ChurnException):
    """Raised when an error occurs during feature engineering."""
    pass

class ModelInferenceError(ChurnException):
    """Raised when an error occurs during prediction/inference."""
    pass
