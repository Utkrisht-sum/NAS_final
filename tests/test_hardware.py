import pytest
import sys
import os
import unittest.mock as mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../micronas')))

# Mock psutil
sys.modules['psutil'] = mock.MagicMock()

from utils.hardware import get_system_info, get_adaptive_nas_config, scale_value

def test_scale_value():
    assert scale_value(0.0, 10, 20) == 10
    assert scale_value(1.0, 10, 20) == 20
    assert scale_value(0.5, 10, 20) == 15

def test_get_system_info():
    info = get_system_info()
    assert "cpu_cores" in info
    assert "ram_gb" in info
    assert "gpu_available" in info
    assert "gpu_memory_gb" in info

def test_adaptive_config_low_end_cpu():
    sys_info = {
        "cpu_cores": 4,
        "ram_gb": 8,
        "gpu_available": False,
        "gpu_memory_gb": 0.0
    }
    config = get_adaptive_nas_config(sys_info)
    # Score = 0.5 * (4/16) + 0.5 * (8/32) = 0.5 * 0.25 + 0.5 * 0.25 = 0.25
    assert config["hardware_score"] == 0.25
    assert config["batch_size"] == 16
    assert config["epochs"] == scale_value(0.25, 5, 15)
    assert config["population"] == scale_value(0.25, 5, 15)
    assert config["generations"] == scale_value(0.25, 3, 8)

def test_adaptive_config_high_end_cpu():
    sys_info = {
        "cpu_cores": 16,
        "ram_gb": 32,
        "gpu_available": False,
        "gpu_memory_gb": 0.0
    }
    config = get_adaptive_nas_config(sys_info)
    # Score = 0.5 * (1.0) + 0.5 * (1.0) = 1.0
    assert config["hardware_score"] == 1.0
    assert config["batch_size"] == 32
    assert config["epochs"] == 30
    assert config["population"] == 30
    assert config["generations"] == 15

def test_adaptive_config_gpu():
    sys_info = {
        "cpu_cores": 16,
        "ram_gb": 32,
        "gpu_available": True,
        "gpu_memory_gb": 24.0
    }
    config = get_adaptive_nas_config(sys_info)
    # Score = 0.3 * (1.0) + 0.3 * (1.0) + 0.4 * (1.0) = 1.0
    assert config["hardware_score"] == 1.0
    assert config["batch_size"] == 256
    assert config["epochs"] == 60
    assert config["population"] == 60
    assert config["generations"] == 30
