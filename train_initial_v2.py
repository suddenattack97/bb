"""
TCN V2 분류 모델 학습 — Triple Barrier + Heikin-Ashi + 알파 피처

변경점:
1. 타겟: Triple Barrier (익절선/손절선/시간) → 0=하락, 1=횡보, 2=상승
2. 입력: Heikin-Ashi 캔들 + 변동성 피처
3. 손실: 방향성 가중 CrossEntropy (up↔down 혼동 시 5배 페널티)
4. 선택: OI, Funding, CVD (데이터 있으면 추가)

실행: python -u train_initial_v2.py  (또는 python train_initial_v2.py)
출력: tcn_v2_model.pth, scaler_v2.npy
"""
import sys
import numpy as np

# 콘솔 실시간 출력
def _log(*a, **kw):
    kw.setdefault('flush', True)
    print(*a, **kw)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model_v2 import TCNClassifier

try:
    import pandas as pd
    import pandas_ta as ta
except ImportError as e:
    raise SystemExit(f"pandas / pandas_ta 필요: pip install pandas pandas_ta — {e}")


# ─────────────────────────────────────────────
#  Triple Barrier 라벨 생성
# ─────────────────────────────────────────────
def triple_barrier_labels(
    df: "pd.DataFrame",
    tp_pct: float = 0.005,   # 익절 0.5%
    sl_pct: float = 0.005,   # 손절 0.5%
    barrier_minutes: int = 5,
    progress_fn=None,       # progress_fn(완료수, 전체수, 퍼센트) 호출
) -> np.ndarray:
    """
    각 행(시점)에서 barrier_minutes 뒤까지의 OHLC를 보고
    익절선/손절선/시간 중 먼저 닿은 것으로 라벨 결정.
    반환: 0=하락(손절), 1=횡보, 2=상승(익절)
    """
    n = len(df)
    total = n - barrier_minutes
    step = max(1, total // 20)  # 약 5%마다 진행률 출력
    labels = np.full(n, 1, dtype=np.int64)  # 기본값 횡보

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values  # iloc 루프 제거용

    for i in range(total):
        entry = closes[i]
        upper = entry * (1 + tp_pct)
        lower = entry * (1 - sl_pct)

        hit_upper = False
        hit_lower = False

        for j in range(1, barrier_minutes + 1):
            h = highs[i + j]
            l = lows[i + j]
            o = opens[i + j]
            # 동일 봉에 상·하단 동시 터치 시: 시가 기준 판단
            if h >= upper and l <= lower:
                if o >= (upper + lower) / 2:
                    hit_upper = True
                else:
                    hit_lower = True
                break
            if h >= upper:
                hit_upper = True
                break
            if l <= lower:
                hit_lower = True
                break

        if hit_upper:
            labels[i] = 2  # 상승
        elif hit_lower:
            labels[i] = 0  # 하락
        else:
            labels[i] = 1  # 횡보

        if progress_fn and (i + 1) % step == 0:
            progress_fn(i + 1, total, 100 * (i + 1) / total)

    if progress_fn and total > 0 and total % step != 0:
        progress_fn(total, total, 100.0)
    return labels


# ─────────────────────────────────────────────
#  Heikin-Ashi 변환
# ─────────────────────────────────────────────
def heikin_ashi(df: "pd.DataFrame") -> "pd.DataFrame":
    """OHLC → Heikin-Ashi OHLC"""
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open = pd.Series(index=df.index, dtype=float)
    ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    ha_high = pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df["low"], ha_open, ha_close], axis=1).min(axis=1)
    return pd.DataFrame({
        "open": ha_open,
        "high": ha_high,
        "low": ha_low,
        "close": ha_close,
        "volume": df["volume"],
    }, index=df.index)


# ─────────────────────────────────────────────
#  방향성 가중 CrossEntropy
# ─────────────────────────────────────────────
class DirectionalLoss(nn.Module):
    """방향(up↔down)을 틀릴 때 더 큰 페널티"""

    def __init__(self, wrong_direction_weight: float = 5.0):
        super().__init__()
        self.wrong_direction_weight = wrong_direction_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # targets: 0=down, 1=flat, 2=up
        n = logits.size(0)
        weights = torch.ones_like(targets, dtype=torch.float32, device=logits.device)
        pred_class = logits.argmax(dim=1)
        # up(2) vs down(0) 혼동 = 최대 페널티
        wrong_dir = ((pred_class == 0) & (targets == 2)) | ((pred_class == 2) & (targets == 0))
        weights[wrong_dir] = self.wrong_direction_weight
        return F.cross_entropy(logits, targets, reduction="none") * weights


# ─────────────────────────────────────────────
#  데이터셋
# ─────────────────────────────────────────────
class TripleBarrierDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        seq_len: int = 60,
        barrier_minutes: int = 5,
        use_heikin_ashi: bool = True,
        use_oi_funding: bool = False,
        oi_csv: str = None,
        funding_csv: str = None,
    ):
        _log("V2 데이터셋 로드 중...")
        # 2022년 이후만 메모리에 로드 (과거 데이터는 시장 구조가 달라 제외)
        df = pd.read_csv(csv_file, index_col="timestamp", parse_dates=True)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        before = len(df)
        df = df.loc["2022-01-01":]
        _log(f"  CSV 로드: {before:,}행 → 2022년 이후 {len(df):,}행 사용")

        orig_volume = df["volume"].copy()
        if use_heikin_ashi:
            _log("  Heikin-Ashi 변환 중...")
            ha = heikin_ashi(df.copy())
            df = ha.copy()
            df["volume"] = orig_volume
            _log("  Heikin-Ashi 완료")

        # Heikin-Ashi 기준 피처
        _log("  피처 계산 중 (log_return, volatility, RSI...)")
        df["log_return"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)
        df["volatility"] = np.log((df["high"] - df["low"] + 1e-8) / (df["close"] + 1e-8) + 1e-8).fillna(0)
        df["rsi_14"] = ta.rsi(df["close"], length=14).fillna(50)
        typical = (df["high"] + df["low"] + df["close"]) / 3
        cum_tpv = (typical * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum()
        df["vwap"] = pd.Series(np.where(cum_vol > 0, cum_tpv / cum_vol, typical), index=df.index).ffill().bfill()
        bb_mid = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std().fillna(0.01)
        df["bb_width"] = ((bb_mid + 2 * bb_std) - (bb_mid - 2 * bb_std)) / df["close"].fillna(0.02)
        df["bb_position"] = ((df["close"] - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-8)).clip(0, 2).fillna(0.5)
        prev_close = df["close"].shift(1).fillna(df["close"])
        tr = np.maximum(df["high"] - df["low"], np.maximum(np.abs(df["high"] - prev_close), np.abs(df["low"] - prev_close)))
        df["atr_14"] = (tr.rolling(14).mean() / df["close"]).ffill().bfill().fillna(0.005)

        # CVD 대리: (close-open) * volume 의 누적 (방향성 거래량)
        df["directional_vol"] = (df["close"] - df["open"]) * df["volume"]
        df["cvd_proxy"] = df["directional_vol"].rolling(20).sum().fillna(0) / (df["volume"].rolling(20).sum() + 1e-8)

        # OI / Funding (파일 있으면 병합)
        if use_oi_funding and oi_csv:
            try:
                oi_df = pd.read_csv(oi_csv, index_col=0, parse_dates=True)
                oi_df = oi_df.reindex(df.index, method="ffill").fillna(0)
                df["oi_change"] = oi_df.iloc[:, 0].pct_change().fillna(0)
            except Exception:
                df["oi_change"] = 0.0
        else:
            df["oi_change"] = 0.0

        if use_oi_funding and funding_csv:
            try:
                fr_df = pd.read_csv(funding_csv, index_col=0, parse_dates=True)
                fr_df = fr_df.reindex(df.index, method="ffill").fillna(0)
                df["funding_rate"] = fr_df.iloc[:, 0]
            except Exception:
                df["funding_rate"] = 0.0
        else:
            df["funding_rate"] = 0.0

        df = df.dropna()
        _log("  피처 계산 완료")

        # Triple Barrier 라벨
        def _on_progress(done, total, pct):
            _log(f"  Triple Barrier 라벨: {done:,} / {total:,} ({pct:.0f}%)")
        _log("  Triple Barrier 라벨 계산 중...")
        labels = triple_barrier_labels(df, barrier_minutes=barrier_minutes, progress_fn=_on_progress)
        df["label"] = labels

        self.seq_len = seq_len
        self.barrier_minutes = barrier_minutes

        features = [
            "log_return", "volatility", "rsi_14", "vwap", "volume",
            "bb_width", "bb_position", "atr_14", "cvd_proxy",
            "oi_change", "funding_rate",
        ]
        self.feature_names = [f for f in features if f in df.columns]
        raw = df[self.feature_names].values.astype(np.float32)

        self.mean = raw.mean(axis=0)
        self.std = raw.std(axis=0) + 1e-8
        self.data_norm = (raw - self.mean) / self.std
        self.labels = df["label"].values.astype(np.int64)

        valid_end = len(df) - seq_len - barrier_minutes
        self.valid_len = max(0, valid_end)
        _log(f"데이터셋 준비 완료: {self.valid_len} 샘플, 피처 {len(self.feature_names)}개")

    def __len__(self) -> int:
        return self.valid_len

    def __getitem__(self, idx: int):
        x = self.data_norm[idx : idx + self.seq_len]
        y = self.labels[idx + self.seq_len]
        return torch.tensor(x), torch.tensor(y, dtype=torch.long)


# ─────────────────────────────────────────────
#  학습
# ─────────────────────────────────────────────
def train_v2(
    csv_file: str = None,
    oi_csv: str = "BTC_futures_oi.csv",
    funding_csv: str = "BTC_funding_rate.csv",
    epochs: int = 5,
    barrier_minutes: int = 5,
):
    from pathlib import Path
    # 현물 OHLCV: fetch_data_v2는 BTC_all_1m_v2.csv, fetch_data는 BTC_all_1m.csv
    if csv_file is None:
        csv_file = "BTC_all_1m_v2.csv" if Path("BTC_all_1m_v2.csv").exists() else "BTC_all_1m.csv"

    # OI/Funding 파일 있으면 자동 병합
    use_oi = Path(oi_csv).exists()
    use_funding = Path(funding_csv).exists()
    if use_oi:
        _log(f"OI 피처 사용: {oi_csv}")
    if use_funding:
        _log(f"Funding 피처 사용: {funding_csv}")

    dataset = TripleBarrierDataset(
        csv_file,
        seq_len=60,
        barrier_minutes=barrier_minutes,
        use_heikin_ashi=True,
        use_oi_funding=(use_oi or use_funding),
        oi_csv=oi_csv if use_oi else None,
        funding_csv=funding_csv if use_funding else None,
    )
    num_features = len(dataset.feature_names)

    train_loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCNClassifier(num_features=num_features, num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = DirectionalLoss(wrong_direction_weight=5.0)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

            if i % 500 == 0:
                _log(f"  Epoch [{epoch+1}/{epochs}] Step [{i}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} Acc: {100*correct/max(1,total):.2f}%")

        avg_loss = total_loss / len(train_loader)
        acc = 100 * correct / total
        _log(f"✅ Epoch {epoch+1}/{epochs} 완료 — Loss: {avg_loss:.4f} Acc: {acc:.2f}%")

    torch.save(model.state_dict(), "tcn_v2_model.pth")
    np.save("scaler_v2.npy", {
        "mean": dataset.mean,
        "std": dataset.std,
        "num_features": num_features,
        "feature_names": dataset.feature_names,
        "use_oi_funding": use_oi or use_funding,  # 예측 시 OI/Funding 사용 여부
    })
    _log("🎉 V2 모델 저장 완료: tcn_v2_model.pth, scaler_v2.npy")


if __name__ == "__main__":
    train_v2()
