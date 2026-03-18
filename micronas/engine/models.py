import torch
import torch.nn as nn
from utils.logger import get_logger

logger = get_logger("ModelBuilder")

class DynamicMLP(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, task="classification"):
        super(DynamicMLP, self).__init__()
        self.task = task
        layers = []

        in_features = input_size
        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # Basic regularization
            in_features = out_features

        self.feature_extractor = nn.Sequential(*layers)

        # Output layer
        out_dim = num_classes if task == "classification" else 1
        self.classifier = nn.Linear(in_features, out_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Flatten input in case it's 2D+
        x = x.view(x.size(0), -1)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

class DynamicCNN(nn.Module):
    def __init__(self, input_shape, conv_layers, fc_layers, num_classes, task="classification"):
        super(DynamicCNN, self).__init__()
        self.task = task
        self.input_shape = input_shape  # e.g., (3, 32, 32)

        # Conv layers: list of dicts [{'channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1}]
        layers = []
        in_channels = input_shape[0]

        current_h, current_w = input_shape[1], input_shape[2]

        for idx, config in enumerate(conv_layers):
            out_channels = config['channels']
            k = config['kernel_size']
            s = config.get('stride', 1)
            p = config.get('padding', k // 2)

            # Dimension check and repair
            if current_h < k or current_w < k:
                logger.warning(f"Spatial dimension ({current_h}x{current_w}) too small for kernel {k} at layer {idx}. Repairing to kernel=1.")
                k = 1
                p = 0

            layers.append(nn.Conv2d(in_channels, out_channels, k, stride=s, padding=p))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2, 2))  # Halves dimensions

            in_channels = out_channels
            current_h = (current_h - k + 2*p) // s + 1
            current_w = (current_w - k + 2*p) // s + 1

            current_h = current_h // 2
            current_w = current_w // 2

            # Stop adding layers if dimensions become 0 or negative
            if current_h <= 0 or current_w <= 0:
                logger.warning("Spatial dimensions collapsed to <= 0. Terminating conv block early.")
                # We pop the last maxpool, relu, batchnorm, conv to prevent crash
                layers = layers[:-4]
                break

        # Global Average Pooling replaces delicate spatial tracking and massively reduces params
        self.feature_extractor = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        fc_layers_list = []
        in_features = in_channels  # Output channels from last conv after pooling to 1x1
        for out_features in fc_layers:
            fc_layers_list.append(nn.Linear(in_features, out_features))
            fc_layers_list.append(nn.ReLU())
            fc_layers_list.append(nn.Dropout(0.3))
            in_features = out_features

        self.fc_block = nn.Sequential(*fc_layers_list)

        out_dim = num_classes if task == "classification" else 1
        self.classifier = nn.Linear(in_features, out_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        x = self.classifier(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_memory_mb(model, input_shape, batch_size=32):
    # Very rough estimate for forward pass memory
    params_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    # Try a forward pass to estimate activations memory (if possible)
    # Using float32 for dummy
    if len(input_shape) == 1:
        dummy = torch.zeros(batch_size, input_shape[0])
    else:
        dummy = torch.zeros(batch_size, *input_shape)

    try:
        with torch.no_grad():
            out = model(dummy)
        # Assuming intermediate activations roughly equal to output size * layers (very rough heuristic)
        acts_mem = (out.numel() * out.element_size() * 10) / (1024 * 1024)
        total_mem = params_mem + acts_mem
        return total_mem
    except:
        return params_mem * 5 # fallback multiplier
