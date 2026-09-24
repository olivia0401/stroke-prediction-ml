"""Project-specific exception types."""


class DataLoadError(Exception):
    """Raised when the input data cannot be loaded."""
    pass


class ModelNotFittedError(ValueError):
    """Raised when prediction is attempted before a model is trained."""
    pass
