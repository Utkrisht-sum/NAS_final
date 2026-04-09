import logging
import os

def get_logger(name):
    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler (Optional logging security: no raw user input directly)
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "micronas.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Prevent propagation to root logger to avoid duplicates
        logger.propagate = False

    return logger
