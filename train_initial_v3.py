"""
TCN V3 분류 모델 학습 — 클래스 균형 + 좁은 Triple Barrier

V2 대비 변경:
1. Triple Barrier 익절/손절 폭 축소 (0.5% → 0.25%): 횡보(0) 감소, 상승/하락 비율 증가
2. 클래스 가중치: 소수 클래스(상승/하락)에 더 큰 페널티 → "0만 찍기" 방지
3. 라벨별 Acc 출력: 하락/횡보/상승 각각 정확도 확인
4. 출력: tcn_v3_model.pth, scaler_v3.npy (V2와 별도 저장)

실행: python -u train_initial_v3.py (V2와 병렬로 학습: 터미널1 python train_initial_v2.py, 터미널2 python train_initial_v3.py)
또는 터미널 2개에서 각각 실행
"""
import sys
import numpy as np

def _log(*a, **kw):
    kw.setdefault('flush', True)
    print(*a, **kw)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from model_v2 import TCNClassifier

try:
    import pandas as pd
    import pandas_ta as ta
except ImportError as e:
    raise SystemExit(f"pandas / pandas_ta 필요: pip install pandas pandas_ta — {e}")


def triple_barrier_labels(
    df: "pd.DataFrame",
    tp_pct: float = 0.0025,  # V3: 0.25% (좁게 → 상승/하락 라벨 증가)
    sl_pct: float = 0.0025,
    barrier_minutes: int = 5,
    progress_fn=None,
) -> np.ndarray:
    """반환: 0=하락, 1=횡보, 2=상승"""
    n = len(df)
    total = n - barrier_minutes
    step = max(1, total // 20)
    labels = np.full(n, 1, dtype=np.int64)
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values

    for i in range(total):
        entry = closes[i]
        upper = entry * (1 + tp_pct)
        lower = entry * (1 - sl_pct)
        hit_upper = False
        hit_lower = False

        for j in range(1, barrier_minutes + 1):
            h, l, o = highs[i + j], lows[i + j], opens[i + j]
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
            labels[i] = 2
        elif hit_lower:
            labels[i] = 0
        else:
            labels[i] = 1

        if progress_fn and (i + 1) % step == 0:
            progress_fn(i + 1, total, 100 * (i + 1) / total)

    if progress_fn and total > 0 and total % step != 0:
        progress_fn(total, total, 100.0)
    return labels


def heikin_ashi(df: "pd.DataFrame") -> "pd.DataFrame":
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open = pd.Series(index=df.index, dtype=float)
    ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    ha_high = pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df["low"], ha_open, ha_close], axis=1).min(axis=1)
    return pd.DataFrame({
        "open": ha_open, "high": ha_high, "low": ha_low, "close": ha_close,
        "volume": df["volume"],
    }, index=df.index)


class DirectionalLossWithClassWeights(nn.Module):
    """클래스 가중치 + 방향 혼동 가중치"""

    def __init__(self, class_weights: torch.Tensor, wrong_direction_weight: float = 5.0):
        super().__init__()
        self.class_weights = class_weights
        self.wrong_direction_weight = wrong_direction_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = self.class_weights[targets]
        pred_class = logits.argmax(dim=1)
        wrong_dir = ((pred_class == 0) & (targets == 2)) | ((pred_class == 2) & (targets == 0))
        mult = torch.where(wrong_dir, torch.full_like(weights, self.wrong_direction_weight), torch.ones_like(weights))
        weights = weights * mult
        return F.cross_entropy(logits, targets, reduction="none") * weights


class TripleBarrierDatasetV3(Dataset):
    """V3: tp_pct/sl_pct 파라미터 추가 (좁은 배리어)"""
    def __init__(
        self,
        csv_file: str,
        seq_len: int = 60,
        barrier_minutes: int = 5,
        tp_pct: float = 0.0025,
        sl_pct: float = 0.0025,
        use_heikin_ashi: bool = True,
        use_oi_funding: bool = False,
        oi_csv: str = None,
        funding_csv: str = None,
    ):
        _log(f"[V3] 데이터셋 로드 중 (tp={tp_pct*100:.2f}%, sl={sl_pct*100:.2f}%)...")
        df = pd.read_csv(csv_file, index_col="timestamp", parse_dates=True)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        df = df.loc["2022-01-01":]
        _log(f"  CSV: 2022년 이후 {len(df):,}행")

        orig_volume = df["volume"].copy()
        if use_heikin_ashi:
            ha = heikin_ashi(df.copy())
            df = ha.copy()
            df["volume"] = orig_volume

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
        df["directional_vol"] = (df["close"] - df["open"]) * df["volume"]
        df["cvd_proxy"] = df["directional_vol"].rolling(20).sum().fillna(0) / (df["volume"].rolling(20).sum() + 1e-8)

        df["oi_change"] = 0.0
        df["funding_rate"] = 0.0
        if use_oi_funding and oi_csv:
            try:
                oi_df = pd.read_csv(oi_csv, index_col=0, parse_dates=True)
                oi_df = oi_df.reindex(df.index, method="ffill").fillna(0)
                df["oi_change"] = oi_df.iloc[:, 0].pct_change().fillna(0)
            except Exception:
                pass
        if use_oi_funding and funding_csv:
            try:
                fr_df = pd.read_csv(funding_csv, index_col=0, parse_dates=True)
                fr_df = fr_df.reindex(df.index, method="ffill").fillna(0)
                df["funding_rate"] = fr_df.iloc[:, 0]
            except Exception:
                pass

        df = df.dropna()

        labels = triple_barrier_labels(df, tp_pct=tp_pct, sl_pct=sl_pct, barrier_minutes=barrier_minutes, progress_fn=lambda d,t,p: _log(f"  TB: {d:,}/{t:,} ({p:.0f}%)"))
        df["label"] = labels

        features = ["log_return", "volatility", "rsi_14", "vwap", "volume", "bb_width", "bb_position", "atr_14", "cvd_proxy", "oi_change", "funding_rate"]
        self.feature_names = [f for f in features if f in df.columns]
        raw = df[self.feature_names].values.astype(np.float32)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        raw = np.clip(raw, -1e6, 1e6)

        self.mean = raw.mean(axis=0)
        self.std = raw.std(axis=0) + 1e-8
        self.data_norm = np.nan_to_num((raw - self.mean) / self.std, nan=0.0, posinf=0.0, neginf=0.0)
        self.labels = df["label"].values.astype(np.int64)

        self.seq_len = seq_len
        valid_end = len(df) - seq_len - barrier_minutes
        self.valid_len = max(0, valid_end)

        # 라벨 분포 출력 (0=하락, 1=횡보, 2=상승)
        unique, counts = np.unique(self.labels[seq_len:seq_len+valid_end], return_counts=True)
        dist = dict(zip(unique, counts))
        _log(f"  라벨 분포 — 하락(0): {dist.get(0,0):,} ({100*dist.get(0,0)/(valid_end+1e-8):.1f}%)  "
             f"횡보(1): {dist.get(1,0):,} ({100*dist.get(1,0)/(valid_end+1e-8):.1f}%)  "
             f"상승(2): {dist.get(2,0):,} ({100*dist.get(2,0)/(valid_end+1e-8):.1f}%)")
        _log(f"  데이터셋: {self.valid_len} 샘플, 피처 {len(self.feature_names)}개")

    def __len__(self) -> int:
        return self.valid_len

    def __getitem__(self, idx: int):
        x = self.data_norm[idx : idx + self.seq_len]
        y = self.labels[idx + self.seq_len]
        return torch.tensor(x), torch.tensor(y, dtype=torch.long)


def train_v3(
    csv_file: str = None,
    oi_csv: str = "BTC_futures_oi.csv",
    funding_csv: str = "BTC_funding_rate.csv",
    epochs: int = 5,
    barrier_minutes: int = 5,
    tp_pct: float = 0.0025,
    sl_pct: float = 0.0025,
):
    from pathlib import Path
    if csv_file is None:
        csv_file = "BTC_all_1m_v2.csv" if Path("BTC_all_1m_v2.csv").exists() else "BTC_all_1m.csv"

    use_oi = Path(oi_csv).exists()
    use_funding = Path(funding_csv).exists()

    dataset = TripleBarrierDatasetV3(
        csv_file,
        seq_len=60,
        barrier_minutes=barrier_minutes,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        use_heikin_ashi=True,
        use_oi_funding=(use_oi or use_funding),
        oi_csv=oi_csv if use_oi else None,
        funding_csv=funding_csv if use_funding else None,
    )
    num_features = len(dataset.feature_names)
    labels = dataset.labels[dataset.seq_len : dataset.seq_len + dataset.valid_len]

    # 클래스 가중치: 소수 클래스에 높은 가중치 (횡보만 찍기 방지)
    unique, counts = np.unique(labels, return_counts=True)
    count_dict = dict(zip(unique, counts))
    total = len(labels)
    weights = np.ones(3)
    for c in range(3):
        n = count_dict.get(c, 1)
        weights[c] = total / (3 * max(n, 1))  # 역빈도
    class_weights = torch.tensor(weights.astype(np.float32))

    sampler = WeightedRandomSampler(
        weights=np.array([weights[y] for y in labels]),
        num_samples=len(labels),
    )

    train_loader = DataLoader(
        dataset,
        batch_size=512,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCNClassifier(num_features=num_features, num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = DirectionalLossWithClassWeights(class_weights.to(device), wrong_direction_weight=5.0)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0
        correct_by_class = [0, 0, 0]
        total_by_class = [0, 0, 0]

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y).mean()
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            pred = logits.argmax(1)
            total_loss += loss.item()
            correct += (pred == y).sum().item()
            total += y.size(0)
            n_batches += 1

            for c in range(3):
                mask = y == c
                if mask.any():
                    total_by_class[c] += mask.sum().item()
                    correct_by_class[c] += (pred[mask] == y[mask]).sum().item()

            if i % 500 == 0:
                acc_all = 100 * correct / max(1, total)
                _log(f"  Epoch [{epoch+1}/{epochs}] Step [{i}/{len(train_loader)}] "
                     f"Loss: {loss.item():.4f} Acc: {acc_all:.2f}%")

        avg_loss = total_loss / max(1, n_batches)
        acc = 100 * correct / max(1, total)
        acc_0 = 100 * correct_by_class[0] / max(1, total_by_class[0])
        acc_1 = 100 * correct_by_class[1] / max(1, total_by_class[1])
        acc_2 = 100 * correct_by_class[2] / max(1, total_by_class[2])
        _log(f"✅ Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f} Acc: {acc:.2f}%  "
             f"(하락: {acc_0:.1f}% 횡보: {acc_1:.1f}% 상승: {acc_2:.1f}%)")

    torch.save(model.state_dict(), "tcn_v3_model.pth")
    np.save("scaler_v3.npy", {
        "mean": dataset.mean,
        "std": dataset.std,
        "num_features": num_features,
        "feature_names": dataset.feature_names,
        "use_oi_funding": use_oi or use_funding,
    })
    _log("🎉 V3 모델 저장 완료: tcn_v3_model.pth, scaler_v3.npy")


if __name__ == "__main__":
    train_v3()
