# utils/logger.py

import logging
from datetime import datetime
from utils.config import LOG_FILE_PATH

def setup_logger():
    """Sets up structured logging to a file and the console."""
    
    # 1. Basic configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    
    # 2. File Handler (for structured logging to run_log.txt)
    file_handler = logging.FileHandler(LOG_FILE_PATH, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Use the same format as above
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    
    # 3. Get the root logger and add the file handler
    root_logger = logging.getLogger()
    # Check if the file handler is already present to prevent duplicate logs
    if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == str(LOG_FILE_PATH) for handler in root_logger.handlers):
        root_logger.addHandler(file_handler)
        
    return root_logger

# Initialize the logger
LOGGER = setup_logger()

def log_event(node_name: str, message: str, level=logging.INFO):
    """
    Logs an event with a standardized format including the node name.
    
    Example: log_event("InferenceNode", "Predicted label: Positive | Confidence: 85.0%")
    """
    full_message = f"[{node_name}] {message}"
    if level == logging.INFO:
        LOGGER.info(full_message)
    elif level == logging.WARNING:
        LOGGER.warning(full_message)
    elif level == logging.ERROR:
        LOGGER.error(full_message)