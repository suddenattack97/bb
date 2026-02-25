"""
V2 데이터 수집 — 현물 1분봉 + 선물 OI / Funding Rate

- 현물 OHLCV: BTC_all_1m.csv (fetch_data.py와 동일 또는 이 스크립트에서 통합)
- 선물 OI: BTC_futures_oi.csv (5분봉으로 수집 후 1분에 forward-fill)
- Funding Rate: BTC_funding_rate.csv (8시간 간격, forward-fill)

실행: python fetch_data_v2.py
"""
import ccxt
import pandas as pd
import time
import os
from datetime import datetime

BINANCE_FUTURES = "binanceusdm"  # USDT-M 선물
SYMBOL = "BTC/USDT"
SPOT_CSV = "BTC_all_1m_v2.csv"
OI_CSV = "BTC_futures_oi.csv"
FUNDING_CSV = "BTC_funding_rate.csv"


def fetch_spot_1m():
    """현물 1분봉 (기존 fetch_data.py와 동일 로직)"""
    exchange = ccxt.binance()
    if os.path.exists(SPOT_CSV):
        existing = pd.read_csv(SPOT_CSV, index_col="timestamp", parse_dates=True)
        start_time = int(existing.index[-1].timestamp() * 1000) + 60000
    else:
        start_time = 1502942400000

    end_time = exchange.milliseconds()
    current = start_time
    while current < end_time:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe="1m", since=current, limit=1000)
            if not ohlcv:
                break
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            write_header = not os.path.exists(SPOT_CSV)
            df.to_csv(SPOT_CSV, mode="a", header=write_header)
            current = ohlcv[-1][0] + 60000
            print(f"  [Spot] {df.index[-1]}")
            time.sleep(0.5)
        except Exception as e:
            print(f"  [Spot] 오류: {e}")
            time.sleep(10)
    print(f"✅ 현물 1분봉 저장: {SPOT_CSV}")


def fetch_open_interest():
    """선물 미결제약정 (Binance: 5m, 15m, 30m, 1h, 2h, 4h만 지원)"""
    try:
        exchange = getattr(ccxt, BINANCE_FUTURES)()
    except AttributeError:
        exchange = ccxt.binance({"options": {"defaultType": "future"}})
    exchange.options = {"defaultType": "future"}
    # 5분봉 OI로 수집
    tf = "5m"
    start_time = int(pd.Timestamp("2022-01-01").timestamp() * 1000)
    if os.path.exists(OI_CSV):
        existing = pd.read_csv(OI_CSV, index_col=0, parse_dates=True)
        start_time = int(existing.index[-1].timestamp() * 1000) + 300000  # 5분
    end_time = exchange.milliseconds()
    current = start_time
    all_data = []
    while current < end_time:
        try:
            # fetch_open_interest_history (일부 거래소만 지원)
            oih = exchange.fetch_open_interest_history(SYMBOL, timeframe=tf, since=current, limit=200)
            if not oih:
                break
            for item in oih:
                all_data.append({
                    "timestamp": pd.Timestamp(item["timestamp"], unit="ms"),
                    "openInterest": item.get("openInterestValue", item.get("openInterest", 0)),
                })
            current = oih[-1]["timestamp"] + 300000
            print(f"  [OI] {all_data[-1]['timestamp']}")
            time.sleep(0.5)
        except Exception as e:
            print(f"  [OI] 오류 (일부 거래소는 미지원): {e}")
            break
    if all_data:
        df = pd.DataFrame(all_data).set_index("timestamp").sort_index()
        df.columns = ["value"]  # train_initial_v2 iloc[:,0] 호환
        df.to_csv(OI_CSV)
        print(f"✅ 선물 OI 저장: {OI_CSV}")
    else:
        print("⚠️ OI 데이터 없음 (Binance 제한 가능)")


def fetch_funding_rate():
    """Funding Rate (보통 8시간 간격)"""
    try:
        exchange = getattr(ccxt, BINANCE_FUTURES)()
    except AttributeError:
        exchange = ccxt.binance({"options": {"defaultType": "future"}})
    exchange.options = {"defaultType": "future"}
    start_time = int(pd.Timestamp("2022-01-01").timestamp() * 1000)
    if os.path.exists(FUNDING_CSV):
        existing = pd.read_csv(FUNDING_CSV, index_col=0, parse_dates=True)
        start_time = int(existing.index[-1].timestamp() * 1000) + 8 * 3600 * 1000
    end_time = exchange.milliseconds()
    current = start_time
    all_data = []
    while current < end_time:
        try:
            frh = exchange.fetch_funding_rate_history(SYMBOL, since=current, limit=200)
            if not frh:
                break
            for item in frh:
                all_data.append({
                    "timestamp": pd.Timestamp(item["timestamp"], unit="ms"),
                    "fundingRate": item.get("fundingRate", 0),
                })
            current = frh[-1]["timestamp"] + 8 * 3600 * 1000
            print(f"  [Funding] {all_data[-1]['timestamp']}")
            time.sleep(0.3)
        except Exception as e:
            print(f"  [Funding] 오류: {e}")
            break
    if all_data:
        df = pd.DataFrame(all_data).set_index("timestamp").sort_index()
        df.columns = ["value"]  # train_initial_v2 iloc[:,0] 호환
        df.to_csv(FUNDING_CSV)
        print(f"✅ Funding Rate 저장: {FUNDING_CSV}")
    else:
        print("⚠️ Funding 데이터 없음")


def main():
    print("V2 데이터 수집 시작")
    print("1. 현물 1분봉...")
    fetch_spot_1m()
    print("2. 선물 OI (선택)...")
    fetch_open_interest()
    print("3. Funding Rate (선택)...")
    fetch_funding_rate()
    print("완료")


if __name__ == "__main__":
    main()
