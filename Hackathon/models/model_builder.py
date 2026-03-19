import torch
import torch.nn as nn


def get_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unknown activation: {name}")


class CNNModel(nn.Module):
    def __init__(self, architecture, num_classes=10, in_channels=1):
        super().__init__()

        layers = []
        current_channels = in_channels

        for layer_cfg in architecture:
            out_channels = layer_cfg["out_channels"]
            kernel_size = layer_cfg["kernel"]
            activation = layer_cfg["activation"]
            use_bn = layer_cfg["batch_norm"]
            dropout = layer_cfg["dropout"]

            layers.append(
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                )
            )

            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))

            layers.append(get_activation(activation))

            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))

            layers.append(nn.MaxPool2d(2))

            current_channels = out_channels

        self.feature_extractor = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(current_channels, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def convert_architecture(arch):
    layers = []

    for units in arch.hidden_units:
        layer_cfg = {
            "out_channels": units,
            "kernel": 3,
            "activation": arch.activation,
            "batch_norm": True,
            "dropout": arch.dropout
        }
        layers.append(layer_cfg)

    return layers