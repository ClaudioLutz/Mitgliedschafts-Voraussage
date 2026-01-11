"""
Centralized Logging Configuration for Mitgliedschafts-Voraussage Pipeline
==========================================================================
Provides consistent logging setup across all modules.

Usage:
    from log_utils import setup_logging, get_logger

    # In entry point scripts (scripts that run directly):
    setup_logging(log_prefix="my_script")
    log = get_logger(__name__)

    # In library modules (imported by other scripts):
    from log_utils import get_logger
    log = get_logger(__name__)

Features:
    - Consistent log format with millisecond precision
    - Console + file handlers
    - Idempotent setup (safe to call multiple times)
    - Execution timer decorator
    - Memory usage logging
"""

import logging
import os
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Optional

# ---------------------
# Configuration
# ---------------------
LOG_FORMAT = '%(asctime)s.%(msecs)03d [%(levelname)s] %(funcName)s: %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
DEFAULT_LOG_DIR = Path(".")
DEFAULT_LOG_PREFIX = "pipeline"


def setup_logging(
    log_level: int = logging.DEBUG,
    log_dir: Optional[Path] = None,
    log_prefix: str = DEFAULT_LOG_PREFIX,
    console_level: Optional[int] = None,
    file_level: Optional[int] = None,
) -> logging.Logger:
    """
    Configure logging for the pipeline with console and file handlers.

    This function is idempotent - calling it multiple times is safe.
    If logging is already configured, it returns the existing root logger.

    Parameters
    ----------
    log_level : int
        Base logging level (default: DEBUG)
    log_dir : Path, optional
        Directory for log files (default: current directory)
    log_prefix : str
        Prefix for log filename (default: "pipeline")
    console_level : int, optional
        Override level for console handler (default: same as log_level)
    file_level : int, optional
        Override level for file handler (default: same as log_level)

    Returns
    -------
    logging.Logger
        The configured root logger

    Example
    -------
    >>> setup_logging(log_level=logging.INFO, log_prefix="training")
    >>> log = get_logger(__name__)
    >>> log.info("Starting process")
    """
    root_logger = logging.getLogger()

    # Skip if already configured (idempotent)
    if root_logger.hasHandlers():
        return root_logger

    # Resolve log directory
    log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = log_dir / f"{log_prefix}_{timestamp}.log"

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level or log_level)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level or log_level)
    file_handler.setFormatter(formatter)

    # Configure root logger
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Log startup info
    root_logger.info(f"Logging initialized. Log file: {log_file}")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    This is a convenience wrapper around logging.getLogger that ensures
    consistent behavior across the codebase.

    Parameters
    ----------
    name : str
        Logger name (typically __name__)

    Returns
    -------
    logging.Logger
        The logger instance

    Example
    -------
    >>> log = get_logger(__name__)
    >>> log.info("Processing started")
    """
    return logging.getLogger(name)


# ---------------------
# Helper Decorators
# ---------------------

def log_execution(func):
    """
    Decorator to log function start, end, and duration.

    Logs output shape if result has .shape attribute (numpy/pandas).
    Logs output length if result is a list or tuple.

    Example
    -------
    >>> @log_execution
    ... def process_data(df):
    ...     return df.dropna()
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        log = logging.getLogger(func.__module__)
        log.info(f"START: {func.__name__}")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            # Add metadata about result
            meta = ""
            if hasattr(result, 'shape'):
                meta = f" | Output Shape: {result.shape}"
            elif isinstance(result, (list, tuple)):
                meta = f" | Output Len: {len(result)}"

            log.info(f"FINISHED: {func.__name__} in {duration:.2f}s{meta}")
            return result
        except Exception as e:
            log.error(f"FAILED: {func.__name__} after {time.time() - start_time:.2f}s - {str(e)}")
            raise
    return wrapper


def log_memory_usage(tag: str = "") -> None:
    """
    Log current memory usage of the process.

    Parameters
    ----------
    tag : str
        Label for the memory checkpoint

    Example
    -------
    >>> log_memory_usage("after loading data")
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_gb = process.memory_info().rss / 1024 / 1024 / 1024
        log = logging.getLogger(__name__)
        log.info(f"RAM USAGE [{tag}]: {mem_gb:.2f} GB")
    except ImportError:
        pass  # psutil not available, skip silently
