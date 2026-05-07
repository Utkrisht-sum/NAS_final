import sys
from unittest.mock import MagicMock

# Mock out heavy dependencies that might be missing in this environment
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.transforms'] = MagicMock()
sys.modules['PySide6'] = MagicMock()
sys.modules['PySide6.QtWidgets'] = MagicMock()
sys.modules['PySide6.QtCore'] = MagicMock()
sys.modules['PySide6.QtCharts'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.backends'] = MagicMock()
sys.modules['matplotlib.backends.backend_qt5agg'] = MagicMock()
sys.modules['matplotlib.figure'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['engine.dataset'] = MagicMock()
sys.modules['engine.nas'] = MagicMock()
sys.modules['engine.trainer'] = MagicMock()
sys.modules['engine.export'] = MagicMock()
sys.modules['utils.logger'] = MagicMock()
sys.modules['gui.charts'] = MagicMock()

from micronas.gui.app import calculate_final_epochs

def test_epoch_calculation_logic():
    # Test Case 1: Standard case, no capping
    # (2*3=6, capped to min 15)
    assert calculate_final_epochs(2, 3, 20) == 15

    # Test Case 2: Accurate prompt (multiplier 3), exceeds default 20
    assert calculate_final_epochs(10, 3, 20) == 20

    # Test Case 3: Accurate prompt, user increased max to 50
    assert calculate_final_epochs(10, 3, 50) == 30

    # Test Case 4: High base epochs, user set low max
    assert calculate_final_epochs(20, 1, 10) == 10

    # Test Case 5: Standard case, within range
    assert calculate_final_epochs(6, 3, 30) == 18
