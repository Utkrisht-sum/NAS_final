import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../micronas')))

from engine.prompt_parser import PromptParser

def test_prompt_parser_valid():
    parser = PromptParser()
    weights = parser.parse("Train a fast classifier")
    assert weights["gamma"] > 0.1 # Base is 0.1

    weights = parser.parse("Build a highly accurate model")
    assert weights["epoch_multiplier"] >= 3
    assert weights["dropout_rate"] >= 0.2

def test_prompt_parser_empty():
    parser = PromptParser()
    weights = parser.parse("")
    assert weights["gamma"] == 0.1
    assert weights["epoch_multiplier"] == 1
    assert weights["dropout_rate"] == 0.2

def test_prompt_parser_malformed():
    parser = PromptParser()
    # It should not crash
    weights = parser.parse(None)
    assert weights["gamma"] == 0.1

    weights = parser.parse(12345)
    assert weights["gamma"] == 0.1

def test_prompt_parser_injection():
    parser = PromptParser()
    # It should just treat it as a string and not exec
    weights = parser.parse("__import__('os').system('ls')")
    assert weights["gamma"] == 0.1
