"""
TCN (Temporal Convolutional Network) 모델 정의
server.py와 train_initial.py 에서 공통으로 임포트하여 사용합니다.
"""
import numpy as np


def predict_linear(prices: list, n_future: int = 5) -> list:
    """
    선형(2차) 회귀 기반 단순 가격 추정.
    torch 모델이 없을 때 fallback으로 사용됩니다.
    """
    if len(prices) < 2:
        return [float(prices[-1])] * n_future
    n = min(len(prices), 20)  # 최근 20개 캔들로 회귀
    subset = prices[-n:]
    x = np.arange(n)
    deg = min(2, n - 1)
    coeffs = np.polyfit(x, subset, deg)
    future_x = np.arange(n, n + n_future)
    predicted = np.polyval(coeffs, future_x)
    # 예측값이 현재가에서 ±5% 이내로 클리핑 (비합리적 예측 방지)
    last_price = float(prices[-1])
    clipped = np.clip(predicted, last_price * 0.95, last_price * 1.05)
    return clipped.tolist()


try:
    import torch
    import torch.nn as nn

    class Chomp1d(nn.Module):
        """Causal padding 제거 레이어"""
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
            self._init_weights()

        def _init_weights(self):
            self.conv1.weight.data.normal_(0, 0.01)
            self.conv2.weight.data.normal_(0, 0.01)
            if self.downsample is not None:
                self.downsample.weight.data.normal_(0, 0.01)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)

    class TCNForecaster(nn.Module):
        """
        Temporal Convolutional Network 기반 BTC 1분봉 예측 모델.
        입력: (batch, seq_len=60, num_features=4) — [log_return, rsi_14, vwap, volume]
        출력: (batch, output_steps=5) — 다음 5분 log_return 예측값
        """
        def __init__(self, num_features: int = 4,
                     num_channels: list = None,
                     kernel_size: int = 3,
                     dropout: float = 0.2,
                     output_steps: int = 5):
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
            self.fc = nn.Linear(num_channels[-1], output_steps)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, features) -> transpose to (batch, features, seq_len)
            y = self.network(x.transpose(1, 2))
            return self.fc(y[:, :, -1])

    TORCH_AVAILABLE = True

except ImportError:
    TORCH_AVAILABLE = False
    TCNForecaster = None
