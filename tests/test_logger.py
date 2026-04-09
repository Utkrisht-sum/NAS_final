import pytest
import logging
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../micronas')))

from utils.logger import get_logger

def test_logger_initialization():
    import uuid
    logger = get_logger(f"TestLogger_Init_{uuid.uuid4().hex}")
    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) == 2
    assert logger.level == logging.INFO

def test_logger_duplicate_prevention():
    logger1 = get_logger("DuplicateTest")
    handlers_count_1 = len(logger1.handlers)

    logger2 = get_logger("DuplicateTest")
    handlers_count_2 = len(logger2.handlers)

    assert logger1 is logger2
    assert handlers_count_1 == handlers_count_2
