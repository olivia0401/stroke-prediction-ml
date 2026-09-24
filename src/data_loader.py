"""Load the raw stroke CSV."""
import pandas as pd

from src.exceptions import DataLoadError
from src.logger import setup_logger

logger = setup_logger(__name__)


def load_data(path):
    """Read the CSV at ``path``; wrap any failure in DataLoadError."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise DataLoadError(f"Failed to load {path}: {e}") from e
    logger.info(f"Loaded {len(df)} samples from {path}")
    return df
