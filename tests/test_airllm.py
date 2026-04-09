import pytest
import sys
import os
import unittest.mock as mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../micronas')))

from utils.airllm_adapter import can_use_airllm, generate_explanation

def test_can_use_airllm_no_gpu():
    sys_info = {"gpu_available": False, "gpu_memory_gb": 0.0}
    assert can_use_airllm(sys_info) == False

def test_can_use_airllm_low_vram():
    sys_info = {"gpu_available": True, "gpu_memory_gb": 8.0}
    assert can_use_airllm(sys_info) == False

def test_can_use_airllm_missing_pkg():
    sys_info = {"gpu_available": True, "gpu_memory_gb": 16.0}
    # Mock missing airllm
    with mock.patch.dict(sys.modules, {'airllm': None}):
        assert can_use_airllm(sys_info) == False

def test_can_use_airllm_success():
    sys_info = {"gpu_available": True, "gpu_memory_gb": 24.0}
    mock_airllm = mock.MagicMock()
    with mock.patch.dict(sys.modules, {'airllm': mock_airllm}):
        assert can_use_airllm(sys_info) == True

def test_generate_explanation_fallback():
    sys_info = {"gpu_available": False, "gpu_memory_gb": 0.0}
    explanation = generate_explanation(sys_info)
    assert "AirLLM Optional Module is offline" in explanation

def test_generate_explanation_success():
    sys_info = {"gpu_available": True, "gpu_memory_gb": 16.0}
    mock_airllm = mock.MagicMock()
    with mock.patch.dict(sys.modules, {'airllm': mock_airllm}):
        explanation = generate_explanation(sys_info)
        assert "AirLLM Inference Completed Successfully" in explanation
