import pandas as pd
import pandas_ta as ta
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 이전 답변의 TCNForecaster, TemporalBlock, Chomp1d 클래스는 이곳에 그대로 복사해서 넣으세요.
# ... (TCN 모델 정의부 생략) ...

class BigBinanceDataset(Dataset):
    def __init__(self, csv_file, seq_len=60, pred_len=5):
        print("대규모 CSV 파일 로드 및 지표 연산 중... (시간이 다소 소요됩니다)")
        df = pd.read_csv(csv_file, index_col='timestamp', parse_dates=True)
        
        # 중복제거 및 정렬 (크롤링 시 혹시 모를 중복 방지)
        df = df[~df.index.duplicated(keep='first')].sort_index()
        
        # 피처 엔지니어링
        df['log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        df['rsi_14'] = ta.rsi(df['close'], length=14).fillna(50)
        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume']).fillna(method='bfill')
        
        # 450만 개 전체를 쓰면 RAM이 부족할 수 있으므로, 최근 2~3년치만 잘라서 쓰는 것도 방법입니다.
        # df = df.loc['2022-01-01':] 
        df = df.dropna()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        features = ['log_return', 'rsi_14', 'vwap', 'volume']
        self.data_raw = df[features].values.astype(np.float32)
        
        self.mean = self.data_raw.mean(axis=0)
        self.std = self.data_raw.std(axis=0) + 1e-8
        self.data_norm = (self.data_raw - self.mean) / self.std
        self.target_idx = 0

    def __len__(self):
        return len(self.data_norm) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        x = self.data_norm[idx : idx + self.seq_len]
        y = self.data_raw[idx + self.seq_len : idx + self.seq_len + self.pred_len, self.target_idx]
        return torch.tensor(x), torch.tensor(y)

def train_base_model():
    dataset = BigBinanceDataset("BTC_all_1m.csv")
    train_loader = DataLoader(dataset, batch_size=1024, shuffle=True) # 대규모이므로 배치사이즈를 키움
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TCNForecaster(num_features=4, output_steps=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    epochs = 5
    for epoch in range(epochs):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            if i % 1000 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.6f}")
                
    torch.save(model.state_dict(), "tcn_base_model.pth")
    np.save("scaler.npy", {'mean': dataset.mean, 'std': dataset.std})
    print("✅ 초기 베이스 모델 학습 및 저장 완료!")

if __name__ == "__main__":
    train_base_model()
