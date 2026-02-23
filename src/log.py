import os
import logging
import time

def setup_logger():
    """Configures logging to both file and console."""
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/benchmark_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Create logger
    logger = logging.getLogger("BenchmarkLogger")
    logger.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # File Handler (Continuous writing)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    
    # Console Handler (To see it in terminal)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger