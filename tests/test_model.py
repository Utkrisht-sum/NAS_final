import pytest
import sys
import os
import unittest.mock as mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../micronas')))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    sys.modules['torch'] = mock.MagicMock()
    sys.modules['torch.nn'] = mock.MagicMock()

try:
    import sklearn
except ImportError:
    sys.modules['sklearn'] = mock.MagicMock()
    sys.modules['sklearn.ensemble'] = mock.MagicMock()

try:
    import xgboost
except ImportError:
    sys.modules['xgboost'] = mock.MagicMock()

from engine.models import DynamicCNN

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch is required for model tests")
def test_dynamic_cnn_small_input():
    input_shape = (3, 32, 32)
    conv_layers = [
        {'channels': 16, 'kernel_size': 3},
        {'channels': 32, 'kernel_size': 3}
    ]
    fc_layers = [64]

    model = DynamicCNN(input_shape, conv_layers, fc_layers, num_classes=10)
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    assert output.shape == (1, 10)

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch is required for model tests")
def test_dynamic_cnn_invalid_shapes():
    # An input shape so small that standard conv operations would fail without proper checks
    input_shape = (3, 2, 2)
    conv_layers = [
        {'channels': 16, 'kernel_size': 3},
        {'channels': 32, 'kernel_size': 3},
        {'channels': 64, 'kernel_size': 3}
    ]
    fc_layers = [64]

    model = DynamicCNN(input_shape, conv_layers, fc_layers, num_classes=10)
    dummy_input = torch.randn(1, 3, 2, 2)
    output = model(dummy_input)
    assert output.shape == (1, 10)

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch is required for model tests")
def test_dynamic_cnn_edge_collapse():
    input_shape = (1, 4, 4)
    # Using many blocks to force spatial dimension collapse early
    conv_layers = [{'channels': 16, 'kernel_size': 3} for _ in range(10)]
    fc_layers = [64]

    model = DynamicCNN(input_shape, conv_layers, fc_layers, num_classes=5)
    dummy_input = torch.randn(2, 1, 4, 4)
    output = model(dummy_input)
    assert output.shape == (2, 5)
