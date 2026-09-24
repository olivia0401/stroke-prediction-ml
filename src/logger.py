"""
Centralized logging for the stroke prediction pipeline.
Replaces print() statements with structured logging.
"""

import logging
from pathlib import Path


def setup_logger(name: str):
    """Configure the root logger once (console + logs/training.log) and return a named logger."""
    Path("logs").mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/training.log"),
            logging.StreamHandler()
        ],
    )

    return logging.getLogger(name)
