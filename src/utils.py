"""Utility functions for MindGuard."""

import logging
import random
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import yaml


def get_device() -> torch.device:
    """
    Detect and return the best available compute device.

    Priority: MPS (Apple Silicon) > CUDA > CPU

    Returns:
        torch.device: The selected device.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        path: Path to the config file.

    Returns:
        Dictionary containing configuration.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def setup_logging(
        level: str = "INFO",
        log_file: str = None
) -> logging.Logger:
    """
    Configure and return a logger for MindGuard.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("mindguard")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def risk_label_to_numeric(label: str) -> int:
    """
    Convert risk label to numeric value.

    Args:
        label: Risk label (Low, Medium, High).

    Returns:
        Numeric value (1, 2, or 3).

    Raises:
        ValueError: If label is invalid.
    """
    mapping = {"Low": 1, "Medium": 2, "High": 3}

    if label not in mapping:
        raise ValueError(f"Invalid risk label: {label}. Must be Low, Medium, or High.")

    return mapping[label]


def numeric_to_risk_label(num: int) -> str:
    """
    Convert numeric value to risk label.

    Args:
        num: Numeric value (1, 2, or 3).

    Returns:
        Risk label (Low, Medium, or High).

    Raises:
        ValueError: If number is invalid.
    """
    mapping = {1: "Low", 2: "Medium", 3: "High"}

    if num not in mapping:
        raise ValueError(f"Invalid numeric value: {num}. Must be 1, 2, or 3.")

    return mapping[num]


def ensure_dir(path: str) -> Path:
    """
    Ensure a directory exists, create if it doesn't.

    Args:
        path: Directory path.

    Returns:
        Path object for the directory.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path