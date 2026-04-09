import pytest
import sys
import os
import unittest.mock as mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../micronas')))

from utils.device import get_device

def test_get_device_no_torch():
    # Mock missing torch
    with mock.patch.dict(sys.modules, {'torch': None}):
        assert get_device() == "cpu"

def test_get_device_with_torch_no_cuda():
    mock_torch = mock.MagicMock()
    mock_torch.cuda.is_available.return_value = False
    with mock.patch.dict(sys.modules, {'torch': mock_torch}):
        assert get_device() == "cpu"

def test_get_device_with_cuda():
    mock_torch = mock.MagicMock()
    mock_torch.cuda.is_available.return_value = True
    with mock.patch.dict(sys.modules, {'torch': mock_torch}):
        assert get_device() == "cuda"
