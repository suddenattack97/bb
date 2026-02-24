"""
TCN 베이스 모델 초기 학습 스크립트
BTC_all_1m.csv 에서 피처를 생성하고 TCNForecaster를 학습합니다.
학습 완료 후 tcn_base_model.pth 와 scaler.npy 가 저장됩니다.
이 두 파일이 있으면 server.py 기동 시 자동으로 로드됩니다.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# model.py 에서 TCNForecaster 임포트
from model import TCNForecaster

try:
    import pandas as pd
    import pandas_ta as ta
except ImportError as e:
    raise SystemExit(
        f"pandas / pandas_ta 가 필요합니다: {e}\n"
        "pip install pandas pandas_ta 를 실행하세요."
    )


class BigBinanceDataset(Dataset):
    def __init__(self, csv_file: str, seq_len: int = 60, pred_len: int = 5):
        print("대규모 CSV 파일 로드 및 지표 연산 중... (시간이 다소 소요됩니다)")
        df = pd.read_csv(csv_file, index_col='timestamp', parse_dates=True)

        # 중복 제거 및 정렬
        df = df[~df.index.duplicated(keep='first')].sort_index()

        # 최근 3년치만 사용 (RAM 절약)
        df = df.loc['2022-01-01':]

        # 피처 엔지니어링
        df['log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        df['rsi_14']     = ta.rsi(df['close'], length=14).fillna(50)
        df['vwap']       = ta.vwap(df['high'], df['low'], df['close'], df['volume']).ffill().bfill()

        df = df.dropna()

        self.seq_len  = seq_len
        self.pred_len = pred_len

        features = ['log_return', 'rsi_14', 'vwap', 'volume']
        raw = df[features].values.astype(np.float32)

        # 정규화 파라미터 저장 (server.py 에서 동일하게 사용)
        self.mean = raw.mean(axis=0)
        self.std  = raw.std(axis=0) + 1e-8
        self.data_norm = (raw - self.mean) / self.std

        # 타깃: log_return (인덱스 0)
        self.data_raw  = raw
        self.target_idx = 0
        print(f"데이터셋 준비 완료: {len(self)} 샘플")

    def __len__(self) -> int:
        return len(self.data_norm) - self.seq_len - self.pred_len

    def __getitem__(self, idx: int):
        x = self.data_norm[idx : idx + self.seq_len]
        y = self.data_raw[idx + self.seq_len : idx + self.seq_len + self.pred_len, self.target_idx]
        return torch.tensor(x), torch.tensor(y)


def train_base_model(csv_file: str = "BTC_all_1m.csv", epochs: int = 5):
    dataset     = BigBinanceDataset(csv_file)
    train_loader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"학습 디바이스: {device}")

    model     = TCNForecaster(num_features=4, output_steps=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 500 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] Step [{i}/{len(train_loader)}] "
                      f"Loss: {loss.item():.6f}")

        avg = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1}/{epochs} 완료 — 평균 Loss: {avg:.6f}")

    torch.save(model.state_dict(), "tcn_base_model.pth")
    np.save("scaler.npy", {'mean': dataset.mean, 'std': dataset.std})
    print("🎉 초기 베이스 모델 학습 및 저장 완료! (tcn_base_model.pth, scaler.npy)")


if __name__ == "__main__":
    train_base_model()
