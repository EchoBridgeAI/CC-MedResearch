import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

def setup_logging():
    """Set up logging with file clearing on each run"""
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Use a single log file for all logging
    log_path = os.path.join(log_dir, 'research_assistant.log')
    
    # Clear the log file at the start of each run
    with open(log_path, 'w') as f:
        f.write(f"=== New Session Started at {datetime.now()} ===\n")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    root_logger.handlers = []
    
    # Create file handler with rotation
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=1024 * 1024,  # 1MB
        backupCount=3
    )
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger.addHandler(file_handler)
    
    return root_logger 