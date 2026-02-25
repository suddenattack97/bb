"""
BTC/USDT V2 방향성 예측 웹서버

- Triple Barrier 분류 모델 (TCNClassifier)
- Heikin-Ashi 피처
- 예측: 상승(2)/횡보(1)/하락(0) → 방향 지표

실행: uvicorn server_v2:app --host 0.0.0.0 --port 8001
"""
import asyncio
import base64
import io
import json
import ssl
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "server_v2.txt"
_tee_lock = threading.Lock()


class TeeWriter(io.TextIOBase):
    def __init__(self, stream, name: str):
        self._stream = stream
        self._name = name

    def write(self, data: str) -> int:
        if data:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{ts}] [{self._name}] {data.rstrip()}\n"
            with _tee_lock:
                try:
                    with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
                        f.write(line)
                        f.flush()
                except Exception:
                    pass
        return self._stream.write(data)

    def flush(self):
        self._stream.flush()

    def isatty(self):
        return self._stream.isatty()


sys.stdout = TeeWriter(sys.stdout, "OUT")
sys.stderr = TeeWriter(sys.stderr, "ERR")

import ccxt
import numpy as np
import pandas as pd
import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from model_v2 import TCNClassifier, TORCH_AVAILABLE, predict_direction_fallback

BINANCE_WS_URLS = [
    "wss://stream.binance.com:9443/ws/btcusdt@kline_1m",
    "wss://stream.binance.com:443/ws/btcusdt@kline_1m",
    "wss://data-stream.binance.vision/ws/btcusdt@kline_1m",
]
POLL_INTERVAL_S = 5
MAX_CANDLE_BUFFER = 60
PRED_STEPS = 5
SNAPSHOT_DIR = Path("snapshots")
CANDLES_DIR = Path("data/candles")
PREDICTIONS_DIR = Path("data/predictions_v2")
SNAPSHOT_DIR.mkdir(exist_ok=True)
CANDLES_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


def _candles_file_for_ts(ts: int) -> Path:
    dt = datetime.fromtimestamp(ts)
    return CANDLES_DIR / f"{dt:%Y-%m-%d}_{dt.hour:02d}.json"


def _load_hour_candles(ts: int) -> list:
    p = _candles_file_for_ts(ts)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            pass
    return []


def _load_latest_hour_candles(ts: int) -> tuple:
    for h_offset in range(24):
        check_ts = ts - h_offset * 3600
        candles = _load_hour_candles(check_ts)
        if candles:
            d = datetime.fromtimestamp(check_ts)
            ps = int(datetime(d.year, d.month, d.day, d.hour, 0, 0).timestamp())
            pe = ps + 3600
            return candles, ps, pe
    return [], 0, 0


def _save_candle(candle: dict):
    ts = candle["time"]
    p = _candles_file_for_ts(ts)
    candles = _load_hour_candles(ts)
    found = False
    for i, c in enumerate(candles):
        if c.get("time") == ts:
            candles[i] = candle
            found = True
            break
    if not found:
        candles.append(candle)
    candles.sort(key=lambda x: x["time"])
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(candles, f, ensure_ascii=False)
    except Exception as e:
        print(f"[SAVE] 캔들 저장 실패: {e}")


def _predictions_file_for_ts(ts: int) -> Path:
    dt = datetime.fromtimestamp(ts)
    return PREDICTIONS_DIR / f"{dt:%Y-%m-%d}_{dt.hour:02d}.json"


def _load_hour_predictions(ts: int) -> list:
    p = _predictions_file_for_ts(ts)
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_prediction(origin_time: int, predictions: list, last_close: float, direction: int, confidence: float) -> bool:
    if not predictions:
        return False
    existing = _load_hour_predictions(origin_time)
    last_end = existing[-1]["predictions"][-1]["time"] if existing else 0
    if origin_time < last_end:
        return False
    entry = {
        "origin_time": origin_time,
        "last_close": last_close,
        "direction": direction,
        "confidence": confidence,
        "predictions": predictions,
    }
    existing.append(entry)
    p = _predictions_file_for_ts(origin_time)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[SAVE] 예측 저장 실패: {e}")
        return False


app = FastAPI(title="BTC V2 방향성 예측 서버")

connected_clients = set()
candle_buffer = []
_prev_candle_ts = 0
_last_pred_debug = {}

_model = None
_scaler_mean = None
_scaler_std = None
_feature_names = None


# ─── Heikin-Ashi & V2 피처 (실시간) ─────────────────────────────────────
def _heikin_ashi_from_buffer(buf: list) -> tuple:
    """버퍼로부터 Heikin-Ashi OHLC 계산"""
    o = np.array([c["open"] for c in buf], dtype=np.float64)
    h = np.array([c["high"] for c in buf], dtype=np.float64)
    l = np.array([c["low"] for c in buf], dtype=np.float64)
    c_ = np.array([c["close"] for c in buf], dtype=np.float64)
    v = np.array([c.get("volume", 0) for c in buf], dtype=np.float64)

    ha_c = (o + h + l + c_) / 4
    ha_o = np.zeros_like(ha_c)
    ha_o[0] = (o[0] + c_[0]) / 2
    for i in range(1, len(ha_o)):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2
    ha_h = np.maximum(h, np.maximum(ha_o, ha_c))
    ha_l = np.minimum(l, np.minimum(ha_o, ha_c))
    return ha_o, ha_h, ha_l, ha_c, v


def _compute_v2_features(buf: list) -> Optional[np.ndarray]:
    """V2 피처 벡터 (마지막 seq_len개)"""
    if len(buf) < 20:
        return None
    ha_o, ha_h, ha_l, ha_c, volumes = _heikin_ashi_from_buffer(buf)

    log_ret = np.diff(np.log(ha_c), prepend=np.log(ha_c[0]))
    volatility = np.log((ha_h - ha_l + 1e-8) / (ha_c + 1e-8) + 1e-8)
    delta = np.diff(ha_c, prepend=ha_c[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = np.convolve(gain, np.ones(14) / 14, mode="same")
    avg_l = np.convolve(loss, np.ones(14) / 14, mode="same")
    rs = np.where(avg_l == 0, 100.0, avg_g / (avg_l + 1e-10))
    rsi = 100 - (100 / (1 + rs))
    typical = (ha_h + ha_l + ha_c) / 3
    cum_tpv = np.cumsum(typical * volumes)
    cum_vol = np.cumsum(volumes)
    vwap = np.where(cum_vol > 0, cum_tpv / cum_vol, typical)
    bb_mid = np.convolve(ha_c, np.ones(20) / 20, mode="same")
    bb_std = np.array([np.std(ha_c[max(0, i - 19) : i + 1]) for i in range(len(ha_c))], dtype=np.float64)
    bb_std = np.where(bb_std < 1e-8, 0.01, bb_std)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = np.where(ha_c > 0, (bb_upper - bb_lower) / ha_c, 0.02)
    bb_pos = np.where((bb_upper - bb_lower) > 1e-8, (ha_c - bb_lower) / (bb_upper - bb_lower), 0.5)
    bb_pos = np.clip(bb_pos, 0, 2)
    prev_c = np.roll(ha_c, 1)
    prev_c[0] = ha_c[0]
    tr = np.maximum(ha_h - ha_l, np.maximum(np.abs(ha_h - prev_c), np.abs(ha_l - prev_c)))
    atr = np.convolve(tr, np.ones(14) / 14, mode="same") / np.where(ha_c > 0, ha_c, 1)
    atr = np.where(np.isnan(atr) | (atr < 1e-8), 0.005, atr)
    directional_vol = (ha_c - ha_o) * volumes
    cvd_proxy = pd.Series(directional_vol).rolling(20).sum().fillna(0).values / (pd.Series(volumes).rolling(20).sum().fillna(1).values + 1e-8)
    oi_change = np.zeros(len(ha_c))
    funding = np.zeros(len(ha_c))

    feat_cols = ["log_return", "volatility", "rsi_14", "vwap", "volume", "bb_width", "bb_position", "atr_14", "cvd_proxy", "oi_change", "funding_rate"]
    feats = np.stack([
        log_ret, volatility, rsi, vwap, volumes,
        bb_width, bb_pos, atr, cvd_proxy, oi_change, funding
    ], axis=1).astype(np.float32)
    return feats


def _load_model():
    global _model, _scaler_mean, _scaler_std, _feature_names
    if not TORCH_AVAILABLE:
        print("[V2] PyTorch 없음 → fallback")
        return
    try:
        import torch
        mp, sp = Path("tcn_v2_model.pth"), Path("scaler_v2.npy")
        if not mp.exists() or not sp.exists():
            print("[V2] tcn_v2_model.pth / scaler_v2.npy 없음 → python train_initial_v2.py 실행 필요")
            return
        sc = np.load(str(sp), allow_pickle=True).item()
        nf = int(sc.get("num_features", 11))
        _feature_names = sc.get("feature_names", [])
        _model = TCNClassifier(num_features=nf, num_classes=3)
        _model.load_state_dict(torch.load(str(mp), map_location="cpu"))
        _model.eval()
        _scaler_mean = sc["mean"]
        _scaler_std = sc["std"]
        print(f"[V2] TCNClassifier 로드 완료 (피처 {nf}개)")
    except Exception as e:
        print(f"[V2] 로드 실패: {e}")


_load_model()


def _direction_to_trajectory(last_close: float, direction: int, n_steps: int = 5) -> list:
    """방향을 시각화용 가격 궤적으로 변환 (상승=+0.3%, 하락=-0.3%, 횡보=0)"""
    step_pct = 0.0006 if direction == 2 else (-0.0006 if direction == 0 else 0)
    values = []
    p = last_close
    for _ in range(n_steps):
        p = p * (1 + step_pct)
        values.append(round(float(p), 2))
    return values


def _run_prediction(buf: list) -> Optional[list]:
    global _last_pred_debug
    if len(buf) < 20:
        return None
    last_ts = buf[-1]["time"]
    closes = np.array([c["close"] for c in buf], dtype=np.float64)
    last_close = float(closes[-1])

    feats = _compute_v2_features(buf)
    if feats is None:
        return None

    direction = 1
    confidence = 0.33
    if _model is not None:
        try:
            import torch
            x = feats[-MAX_CANDLE_BUFFER:]
            if _scaler_mean is not None:
                x = (x - _scaler_mean) / (_scaler_std + 1e-8)
            x = torch.tensor(x).unsqueeze(0)
            with torch.no_grad():
                logits = _model(x)
                probs = torch.softmax(logits, dim=1).squeeze().numpy()
                pred_class = int(logits.argmax(dim=1).item())
            direction = pred_class
            confidence = float(probs[pred_class])
            _last_pred_debug = {
                "model": "TCNClassifier",
                "direction": direction,
                "probs": [float(p) for p in probs],
                "origin_time": last_ts,
            }
        except Exception as e:
            print(f"[V2 PRED] 오류: {e}")
            direction = predict_direction_fallback()
    else:
        direction = predict_direction_fallback()
        _last_pred_debug = {"model": "fallback", "direction": direction}

    traj = _direction_to_trajectory(last_close, direction, PRED_STEPS)
    result = []
    for i, v in enumerate(traj):
        result.append({
            "time": last_ts + (i + 1) * 60,
            "value": v,
            "lower": round(v * 0.999, 2),
            "upper": round(v * 1.001, 2),
            "direction": direction,
            "confidence": confidence,
        })
    return result


async def _broadcast(message: dict):
    if not connected_clients:
        return
    payload = json.dumps(message, ensure_ascii=False)
    dead = set()
    for client in list(connected_clients):
        try:
            await client.send_text(payload)
        except Exception:
            dead.add(client)
    connected_clients.difference_update(dead)


async def _handle_candle(candle: dict):
    global _prev_candle_ts
    payload = {
        "time": candle["time"],
        "open": candle["open"],
        "high": candle["high"],
        "low": candle["low"],
        "close": candle["close"],
        "volume": candle.get("volume", 0),
    }
    await _broadcast({"type": "real_time_candle", "candle": payload})

    if candle["is_closed"] and candle["time"] != _prev_candle_ts:
        _prev_candle_ts = candle["time"]
        _save_candle(payload)
        candle_buffer.append(payload)
        if len(candle_buffer) > MAX_CANDLE_BUFFER:
            candle_buffer.pop(0)
        predictions = _run_prediction(candle_buffer)
        if predictions:
            d = predictions[0].get("direction", 1)
            conf = predictions[0].get("confidence", 0.33)
            _save_prediction(candle["time"], predictions, float(candle["close"]), d, conf)
            await _broadcast({
                "type": "prediction",
                "predictions": predictions,
                "origin_time": candle["time"],
            })


def _parse_kline_msg(raw: str) -> Optional[dict]:
    try:
        data = json.loads(raw)
        k = data.get("k") or data
        if not isinstance(k, dict) or "o" not in k:
            return None
        return {
            "time": int(k["t"]) // 1000,
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k["v"]),
            "is_closed": bool(k["x"]),
        }
    except Exception:
        return None


async def _try_ws_url(url: str) -> bool:
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    try:
        print(f"[V2 WS] 연결: {url}")
        async with websockets.connect(url, ssl=ssl_ctx, open_timeout=10, ping_interval=20, ping_timeout=20) as ws:
            print(f"[V2 WS] ✅ 연결 성공")
            async for message in ws:
                candle = _parse_kline_msg(message)
                if candle:
                    await _handle_candle(candle)
        return True
    except Exception as e:
        print(f"[V2 WS] 실패: {e}")
        return False


async def _binance_ws_feed():
    for url in BINANCE_WS_URLS:
        if await _try_ws_url(url):
            return True
    return False


async def _binance_rest_feed():
    exchange = ccxt.binance({"enableRateLimit": True})
    print("[V2 REST] ccxt 폴링 모드")
    loop = asyncio.get_event_loop()
    while True:
        try:
            ohlcv = await loop.run_in_executor(None, lambda: exchange.fetch_ohlcv("BTC/USDT", "1m", limit=3))
        except Exception as e:
            print(f"[V2 REST] 오류: {e}")
            await asyncio.sleep(POLL_INTERVAL_S)
            continue
        for row in ohlcv[:-1]:
            try:
                await _handle_candle({
                    "time": row[0] // 1000,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "is_closed": True,
                })
            except Exception:
                pass
        if ohlcv:
            cur = ohlcv[-1]
            try:
                await _handle_candle({
                    "time": cur[0] // 1000,
                    "open": float(cur[1]),
                    "high": float(cur[2]),
                    "low": float(cur[3]),
                    "close": float(cur[4]),
                    "volume": float(cur[5]),
                    "is_closed": False,
                })
            except Exception:
                pass
        await asyncio.sleep(POLL_INTERVAL_S)


async def _binance_feed():
    while True:
        print("[V2] Binance 연결 시도...")
        if await _binance_ws_feed():
            await asyncio.sleep(3)
        else:
            await _binance_rest_feed()


@app.on_event("startup")
async def _startup():
    asyncio.create_task(_binance_feed())


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.add(ws)
    now_ts = int(datetime.now().timestamp())
    hour_candles, period_start, period_end = _load_latest_hour_candles(now_ts)
    if hour_candles and period_start and period_end:
        try:
            await ws.send_text(json.dumps({
                "type": "hour_history",
                "candles": hour_candles,
                "period_start": period_start,
                "period_end": period_end,
                "is_past": period_end <= now_ts,
            }, ensure_ascii=False))
        except Exception as e:
            print(f"[V2] hour_history 오류: {e}")
    pred_ts = period_start if period_start else now_ts
    hour_predictions = _load_hour_predictions(pred_ts)
    if hour_predictions:
        try:
            await ws.send_text(json.dumps({"type": "prediction_history", "predictions": hour_predictions}, ensure_ascii=False))
        except Exception:
            pass
    try:
        while True:
            await ws.receive_text()
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        connected_clients.discard(ws)


@app.post("/save_snapshot")
async def save_snapshot(request: Request):
    body = await request.json()
    raw = body.get("data", "")
    filename = body.get("filename", "snapshot.png")
    if "," in raw:
        raw = raw.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(raw)
    except Exception:
        return JSONResponse({"error": "base64 실패"}, status_code=400)
    safe = Path(filename).name
    if not safe.endswith(".png"):
        safe += ".png"
    (SNAPSHOT_DIR / safe).write_bytes(img_bytes)
    return {"status": "ok", "filename": safe, "size": len(img_bytes)}


@app.get("/snapshots")
async def list_snapshots():
    files = sorted(SNAPSHOT_DIR.glob("*.png"), key=lambda f: f.stat().st_mtime, reverse=True)
    return {"snapshots": [f.name for f in files[:20]]}


@app.get("/snapshots/{filename}")
async def get_snapshot(filename: str):
    path = SNAPSHOT_DIR / Path(filename).name
    if not path.exists():
        return JSONResponse({"error": "파일 없음"}, status_code=404)
    return FileResponse(str(path), media_type="image/png")


@app.get("/status")
async def status():
    return {
        "clients": len(connected_clients),
        "candle_buffer": len(candle_buffer),
        "model": "TCNClassifier" if _model else "fallback",
        "last_close": candle_buffer[-1]["close"] if candle_buffer else None,
    }


@app.get("/api/model_debug")
async def model_debug():
    return {"model_loaded": _model is not None, **_last_pred_debug}


@app.get("/api/logs")
async def get_logs(tail: int = 1000):
    if not LOG_FILE.exists():
        return {"content": "", "path": str(LOG_FILE)}
    with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    content = "".join(lines[-tail:]) if len(lines) > tail else "".join(lines)
    return {"content": content, "path": str(LOG_FILE), "lines": len(lines)}


@app.get("/api/logs/download")
async def download_logs():
    if not LOG_FILE.exists():
        return JSONResponse({"error": "로그 없음"}, status_code=404)
    return FileResponse(LOG_FILE, filename="server_v2.txt", media_type="text/plain")


@app.get("/")
async def index():
    return FileResponse("index_v2.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_v2:app", host="0.0.0.0", port=8001, reload=False)
