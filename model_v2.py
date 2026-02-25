"""
TCN 분류 모델 (V2) — Triple Barrier 방향성 예측
출력: 3클래스 (0=하락, 1=횡보, 2=상승) → CrossEntropy
"""
import numpy as np

try:
    import torch
    import torch.nn as nn

    class Chomp1d(nn.Module):
        def __init__(self, chomp_size: int):
            super().__init__()
            self.chomp_size = chomp_size

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, :, :-self.chomp_size].contiguous()

    class TemporalBlock(nn.Module):
        def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int,
                     stride: int, dilation: int, padding: int, dropout: float = 0.2):
            super().__init__()
            self.conv1 = nn.utils.weight_norm(
                nn.Conv1d(n_inputs, n_outputs, kernel_size,
                          stride=stride, padding=padding, dilation=dilation)
            )
            self.chomp1 = Chomp1d(padding)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)

            self.conv2 = nn.utils.weight_norm(
                nn.Conv1d(n_outputs, n_outputs, kernel_size,
                          stride=stride, padding=padding, dilation=dilation)
            )
            self.chomp2 = Chomp1d(padding)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)

            self.net = nn.Sequential(
                self.conv1, self.chomp1, self.relu1, self.dropout1,
                self.conv2, self.chomp2, self.relu2, self.dropout2
            )
            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)

    class TCNClassifier(nn.Module):
        """
        TCN 기반 방향성 분류 모델 (Triple Barrier).
        출력: 3클래스 logits (하락=0, 횡보=1, 상승=2)
        """
        def __init__(self, num_features: int,
                     num_channels: list = None,
                     kernel_size: int = 3,
                     dropout: float = 0.2,
                     num_classes: int = 3):
            super().__init__()
            if num_channels is None:
                num_channels = [64, 128, 128]

            layers = []
            for i, out_ch in enumerate(num_channels):
                dilation = 2 ** i
                in_ch = num_features if i == 0 else num_channels[i - 1]
                padding = (kernel_size - 1) * dilation
                layers.append(TemporalBlock(
                    in_ch, out_ch, kernel_size,
                    stride=1, dilation=dilation,
                    padding=padding, dropout=dropout
                ))

            self.network = nn.Sequential(*layers)
            self.fc = nn.Linear(num_channels[-1], num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.network(x.transpose(1, 2))
            return self.fc(y[:, :, -1])

    TORCH_AVAILABLE = True

except ImportError:
    TORCH_AVAILABLE = False
    TCNClassifier = None


def predict_direction_fallback() -> int:
    """모델 없을 때 fallback: 1 (횡보) 반환"""
    return 1
