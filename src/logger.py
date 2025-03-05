#!/usr/bin/env python3
"""
Logger module for the AI system.
Provides a centralized logging configuration to ensure consistent log formatting
across all modules.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import datetime

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = os.path.join(PROJECT_ROOT, "log")

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Log file naming convention
def get_log_filename():
    """Generate log filename with current date"""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    return f"agent_{today}.log"

# Log formatter
LOG_FORMAT = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def setup_logger(name, level=logging.INFO):
    """
    Setup and return a logger with the given name and level.
    
    Args:
        name (str): The name for the logger (typically __name__ from the calling module)
        level (int or str): The logging level (default: logging.INFO)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())
        
    logger = logging.getLogger(name)
    
    # If the logger has already been configured, return it
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Create a file handler that writes to the log directory
    log_file_path = os.path.join(LOG_DIR, get_log_filename())
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=10485760,  # 10 MB
        backupCount=10
    )
    file_handler.setFormatter(LOG_FORMAT)
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(LOG_FORMAT)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Example usage:
# from src.logger import setup_logger
# logger = setup_logger(__name__)
# logger.info("Application starting")