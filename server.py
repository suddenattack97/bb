"""
BTC/USDT 1분봉 실시간 예측 웹서버

Binance 연결 전략 (순서대로 시도):
  1) websockets 라이브러리  - stream.binance.com:9443  (표준)
  2) websockets 라이브러리  - stream.binance.com:443   (방화벽 우회)
  3) websockets 라이브러리  - data-stream.binance.vision (CDN)
  4) ccxt REST 폴링 fallback - 5초 간격

분봉 데이터: data/candles/YYYY-MM-DD_HH.json 에 시간별로 저장.
웹 접속 시 해당 시간대 파일에서 과거 분봉을 로드해 채운 뒤 실시간 수신 계속.
실행: uvicorn server:app --host 0.0.0.0 --port 8000
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

# Windows CP949 콘솔에서 이모지 출력 오류 방지
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# ─────────────────────────────────────────────
#  로그 파일 Tee (stdout/stderr → 콘솔 + txt 파일)
# ─────────────────────────────────────────────
LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "server.txt"
_tee_lock = threading.Lock()


class TeeWriter(io.TextIOBase):
    """콘솔과 로그 파일에 동시 출력"""

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


_sys_stdout = sys.stdout
_sys_stderr = sys.stderr
sys.stdout = TeeWriter(_sys_stdout, "OUT")
sys.stderr = TeeWriter(_sys_stderr, "ERR")

import ccxt
import numpy as np
import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from model import predict_linear, TORCH_AVAILABLE

# ─────────────────────────────────────────────
#  설정
# ─────────────────────────────────────────────
BINANCE_WS_URLS = [
    "wss://stream.binance.com:9443/ws/btcusdt@kline_1m",
    "wss://stream.binance.com:443/ws/btcusdt@kline_1m",
    "wss://data-stream.binance.vision/ws/btcusdt@kline_1m",
]
BINANCE_SYMBOL    = "BTC/USDT"
POLL_INTERVAL_S   = 5
MAX_CANDLE_BUFFER = 60
PRED_STEPS        = 5
SNAPSHOT_DIR      = Path("snapshots")
CANDLES_DIR       = Path("data/candles")
PREDICTIONS_DIR   = Path("data/predictions")
SNAPSHOT_DIR.mkdir(exist_ok=True)
CANDLES_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


def _candles_file_for_ts(ts: int) -> Path:
    """Unix timestamp -> 저장 파일 경로 (서버 로컬 시간 기준 시간대)"""
    dt = datetime.fromtimestamp(ts)
    return CANDLES_DIR / f"{dt:%Y-%m-%d}_{dt.hour:02d}.json"


def _load_hour_candles(ts: int) -> list[dict]:
    """해당 시간대(ts가 포함된 시)의 분봉 목록 로드. 없으면 []."""
    p = _candles_file_for_ts(ts)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data and isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def _load_latest_hour_candles(ts: int) -> tuple[list[dict], int, int]:
    """
    저장된 분봉 중 가장 최근 데이터 로드. (재시작 후에도 이전 시간대 데이터 표시)
    반환: (candles, period_start, period_end) 또는 ([], 0, 0)
    """
    dt = datetime.fromtimestamp(ts)
    for h_offset in range(0, 24):
        check_ts = ts - h_offset * 3600
        candles = _load_hour_candles(check_ts)
        if candles:
            d = datetime.fromtimestamp(check_ts)
            ps = int(datetime(d.year, d.month, d.day, d.hour, 0, 0).timestamp())
            pe = ps + 3600
            return candles, ps, pe
    return [], 0, 0


def _save_candle(candle: dict):
    """완결된 1분봉을 해당 시간대 파일에 저장 (upsert)."""
    ts = candle["time"]
    p = _candles_file_for_ts(ts)
    candles = _load_hour_candles(ts)
    # time 기준 upsert
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
    """Unix timestamp -> 예측 저장 파일 경로 (서버 로컬 시간 기준 시간대)"""
    dt = datetime.fromtimestamp(ts)
    return PREDICTIONS_DIR / f"{dt:%Y-%m-%d}_{dt.hour:02d}.json"


def _load_hour_predictions(ts: int) -> list[dict]:
    """해당 시간대의 예측 히스토리 로드. 없으면 []."""
    p = _predictions_file_for_ts(ts)
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_prediction(origin_time: int, predictions: list[dict], last_close: float) -> bool:
    """
    예측을 파일에 저장. 클라이언트와 동일하게 '예측 끝 시각 도달 시에만' 추가.
    반환: 실제로 저장했으면 True.
    """
    if not predictions:
        return False
    existing = _load_hour_predictions(origin_time)
    # 마지막 예측의 끝 시각
    last_end = existing[-1]["predictions"][-1]["time"] if existing else 0
    if origin_time < last_end:
        return False  # 아직 이전 예측 구간 도달 전
    entry = {
        "origin_time": origin_time,
        "last_close": last_close,
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


app = FastAPI(title="BTC 1분봉 예측 서버")

# ─────────────────────────────────────────────
#  전역 상태
# ─────────────────────────────────────────────
connected_clients: set[WebSocket] = set()
candle_buffer: list[dict]         = []   # 완결 캔들 저장 (최대 60개)
_prev_candle_ts: int              = 0    # 마지막 완결 캔들 타임스탬프 (중복 예측 방지)
_last_pred_debug: dict            = {}  # 검증용: 마지막 예측 입력/출력

# ─────────────────────────────────────────────
#  모델 로드 (선택적)
# ─────────────────────────────────────────────
_model       = None
_scaler_mean: Optional[np.ndarray] = None
_scaler_std:  Optional[np.ndarray] = None

NUM_FEATURES = 7

def _load_model():
    global _model, _scaler_mean, _scaler_std, NUM_FEATURES
    if not TORCH_AVAILABLE:
        print("[MODEL] PyTorch 없음 → 선형 추정 fallback 사용")
        return
    try:
        import torch
        from model import TCNForecaster
        mp, sp = Path("tcn_base_model.pth"), Path("scaler.npy")
        if mp.exists() and sp.exists():
            sc = np.load(str(sp), allow_pickle=True).item()
            nf = int(sc.get("num_features", 4))
            NUM_FEATURES = nf
            m = TCNForecaster(num_features=nf, output_steps=PRED_STEPS)
            m.load_state_dict(torch.load(str(mp), map_location="cpu"))
            m.eval()
            _model = m
            _scaler_mean, _scaler_std = sc["mean"], sc["std"]
            print(f"[MODEL] TCN 모델 로드 완료 (피처 {nf}개)")
        else:
            print("[MODEL] 모델 파일 없음 → 선형 추정 fallback 사용")
    except Exception as e:
        print(f"[MODEL] 로드 실패: {e} → 선형 추정 fallback 사용")

_load_model()

# ─────────────────────────────────────────────
#  예측 로직 (기술적 지표 + 볼린저 구간)
# ─────────────────────────────────────────────
def _run_prediction(buf: list[dict]) -> Optional[list[dict]]:
    global _last_pred_debug
    if len(buf) < 5:
        return None
    highs   = np.array([c["high"]   for c in buf], dtype=np.float64)
    lows    = np.array([c["low"]    for c in buf], dtype=np.float64)
    closes  = np.array([c["close"]  for c in buf], dtype=np.float64)
    volumes = np.array([c["volume"] for c in buf], dtype=np.float64)
    last_ts = buf[-1]["time"]

    # 공통 지표
    log_ret = np.diff(np.log(closes), prepend=np.log(closes[0]))
    delta   = np.diff(closes, prepend=closes[0])
    gain    = np.where(delta > 0, delta, 0.0)
    loss    = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.convolve(gain, np.ones(14) / 14, mode="same")
    avg_loss = np.convolve(loss, np.ones(14) / 14, mode="same")
    rs      = np.where(avg_loss == 0, 100.0, avg_gain / (avg_loss + 1e-10))
    rsi     = 100 - (100 / (1 + rs))
    typical = (highs + lows + closes) / 3.0
    cum_tpv = np.cumsum(typical * volumes)
    cum_vol = np.cumsum(volumes)
    vwap    = np.where(cum_vol > 0, cum_tpv / cum_vol, typical)

    if _model is not None:
        try:
            import torch
            if NUM_FEATURES >= 7:
                bb_mid   = np.convolve(closes, np.ones(20) / 20, mode="same")
                bb_std   = np.array([np.std(closes[max(0,i-19):i+1]) for i in range(len(closes))], dtype=np.float64)
                bb_std   = np.where(bb_std < 1e-8, 0.01, bb_std)
                bb_upper = bb_mid + 2 * bb_std
                bb_lower = bb_mid - 2 * bb_std
                bb_width = np.where(closes > 0, (bb_upper - bb_lower) / closes, 0.02)
                bb_pos   = np.where((bb_upper - bb_lower) > 1e-8, (closes - bb_lower) / (bb_upper - bb_lower), 0.5)
                bb_pos   = np.clip(bb_pos, 0, 2)
                tr = np.maximum(highs - lows, np.maximum(np.abs(highs - np.roll(closes, 1)), np.abs(lows - np.roll(closes, 1))))
                tr[0] = highs[0] - lows[0]
                atr_14 = np.convolve(tr, np.ones(14) / 14, mode="same") / np.where(closes > 0, closes, 1)
                atr_14 = np.where(np.isnan(atr_14) | (atr_14 < 1e-8), 0.005, atr_14)
                feats = np.stack([log_ret, rsi, vwap, volumes, bb_width, bb_pos, atr_14], axis=1).astype(np.float32)
            else:
                feats = np.stack([log_ret, rsi, vwap, volumes], axis=1).astype(np.float32)

            if _scaler_mean is not None:
                feats = (feats - _scaler_mean) / (_scaler_std + 1e-8)
            x = torch.tensor(feats[-MAX_CANDLE_BUFFER:]).unsqueeze(0)
            with torch.no_grad():
                pred_lr = _model(x).squeeze().numpy()
            last_c, pred_prices = float(closes[-1]), []
            for lr in pred_lr:
                last_c = float(last_c) * float(np.exp(lr))
                pred_prices.append(last_c)

            _last_pred_debug.update({
                "model": "TCN",
                "num_features": NUM_FEATURES,
                "origin_time": last_ts,
                "input_last_close": float(closes[-1]),
                "pred_log_returns": [float(x) for x in pred_lr],
                "pred_prices": [round(float(p), 2) for p in pred_prices],
                "bb_width_last": float(bb_width[-1]) if NUM_FEATURES >= 7 else None,
            })
        except Exception as e:
            print(f"[PRED] 모델 추론 오류: {e} → fallback")
            pred_prices = predict_linear([c["close"] for c in buf], PRED_STEPS)
            _last_pred_debug.update({"model": "linear_fallback", "error": str(e)})
    else:
        pred_prices = predict_linear([c["close"] for c in buf], PRED_STEPS)
        _last_pred_debug.update({"model": "linear_fallback", "reason": "no_model"})

    # 볼린저 구간 (최근 20봉 변동성 기반)
    bb_mid = np.convolve(closes, np.ones(20) / 20, mode="same")
    bb_std = np.array([np.std(closes[max(0, i - 19):i + 1]) for i in range(len(closes))], dtype=np.float64)
    bb_std = np.where(bb_std < 1e-8, closes * 0.001, bb_std)
    bb_width_arr = (2 * 2 * bb_std) / np.where(closes > 0, closes, 1)
    half_range = float(bb_width_arr[-1]) * 0.5 if len(bb_width_arr) else 0.01

    result = []
    for i, p in enumerate(pred_prices):
        pv = float(p)
        lo = round(pv * (1 - half_range), 2)
        hi = round(pv * (1 + half_range), 2)
        lo, hi = min(lo, pv), max(hi, pv)
        result.append({
            "time": last_ts + (i + 1) * 60,
            "value": round(pv, 2),
            "lower": lo,
            "upper": hi,
        })
    return result

# ─────────────────────────────────────────────
#  WebSocket 브로드캐스트 (프론트엔드 → )
# ─────────────────────────────────────────────
async def _broadcast(message: dict):
    if not connected_clients:
        return
    payload = json.dumps(message, ensure_ascii=False)
    dead: set[WebSocket] = set()
    for client in list(connected_clients):
        try:
            await client.send_text(payload)
        except Exception:
            dead.add(client)
    connected_clients.difference_update(dead)

# ─────────────────────────────────────────────
#  캔들 처리 (Binance → 프론트엔드)
# ─────────────────────────────────────────────
async def _handle_candle(candle: dict):
    """캔들 수신 시 분단위로 전송. 마감 시 파일 저장 + 예측 브로드캐스트."""
    global _prev_candle_ts

    # volume 포함해 전체 전송 (예측/저장에 필요)
    payload = {
        "time": candle["time"],
        "open": candle["open"],
        "high": candle["high"],
        "low": candle["low"],
        "close": candle["close"],
        "volume": candle.get("volume", 0),
    }

    # ① 분단위 캔들 브로드캐스트 (틱이 아닌 1분 단위 - 동일 분은 클라이언트에서 upsert)
    await _broadcast({"type": "real_time_candle", "candle": payload})

    # ② 마감된 1분봉: 파일 저장 + 예측 실행 (중복 방지)
    if candle["is_closed"] and candle["time"] != _prev_candle_ts:
        _prev_candle_ts = candle["time"]
        _save_candle(payload)
        candle_buffer.append(payload)
        if len(candle_buffer) > MAX_CANDLE_BUFFER:
            candle_buffer.pop(0)
        predictions = _run_prediction(candle_buffer)
        if predictions:
            _save_prediction(candle["time"], predictions, float(candle["close"]))
            await _broadcast({
                "type": "prediction",
                "predictions": predictions,
                "origin_time": candle["time"],
            })

def _parse_kline_msg(raw: str) -> Optional[dict]:
    """Binance kline WebSocket 메시지를 캔들 dict로 변환."""
    try:
        data = json.loads(raw)
        k = data.get("k") or data  # 스트림 형식에 따라 처리
        if not isinstance(k, dict) or "o" not in k:
            return None
        return {
            "time":      int(k["t"]) // 1000,
            "open":      float(k["o"]),
            "high":      float(k["h"]),
            "low":       float(k["l"]),
            "close":     float(k["c"]),
            "volume":    float(k["v"]),
            "is_closed": bool(k["x"]),
        }
    except Exception:
        return None

# ─────────────────────────────────────────────
#  Binance WebSocket 연결 (websockets 라이브러리)
# ─────────────────────────────────────────────
async def _try_ws_url(url: str) -> bool:
    """
    단일 URL로 WebSocket 연결을 시도합니다.
    연결 성공 후 정상 루프 시 True 반환.
    연결/수신 오류 시 False 반환.
    """
    ssl_ctx = ssl.create_default_context()
    # 일부 방화벽 환경에서 인증서 검증 실패 시를 대비한 옵션
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode    = ssl.CERT_NONE
    try:
        print(f"[WS] 연결 시도: {url}")
        async with websockets.connect(
            url,
            ssl=ssl_ctx,
            open_timeout=10,
            ping_interval=20,
            ping_timeout=20,
        ) as ws:
            print(f"[WS] ✅ 연결 성공: {url}")
            async for message in ws:
                candle = _parse_kline_msg(message)
                if candle:
                    await _handle_candle(candle)
        return True   # 연결 후 정상 종료
    except Exception as e:
        print(f"[WS] 실패 ({url}): {type(e).__name__}: {e}")
        return False

async def _binance_ws_feed() -> bool:
    """3개 URL을 순서대로 시도. 하나라도 성공하면 True 반환."""
    for url in BINANCE_WS_URLS:
        ok = await _try_ws_url(url)
        if ok:
            return True
    return False

# ─────────────────────────────────────────────
#  REST 폴링 fallback (ccxt, 5초 간격)
# ─────────────────────────────────────────────
async def _binance_rest_feed():
    """WebSocket 모든 URL 실패 시 ccxt REST API로 폴링."""
    exchange = ccxt.binance({"enableRateLimit": True})
    print(f"[REST] ccxt 폴링 모드 시작 (간격: {POLL_INTERVAL_S}s)")
    loop = asyncio.get_event_loop()

    def _fetch():
        # limit=3: 직전 완결 2개 + 현재 진행 1개
        return exchange.fetch_ohlcv(BINANCE_SYMBOL, "1m", limit=3)

    while True:
        try:
            ohlcv = await loop.run_in_executor(None, _fetch)
        except Exception as e:
            print(f"[REST] fetch 오류: {e}")
            await asyncio.sleep(POLL_INTERVAL_S)
            continue

        # 완결 캔들들 처리 (마지막 제외)
        for row in ohlcv[:-1]:
            try:
                await _handle_candle({
                    "time":      row[0] // 1000,
                    "open":      float(row[1]),
                    "high":      float(row[2]),
                    "low":       float(row[3]),
                    "close":     float(row[4]),
                    "volume":    float(row[5]),
                    "is_closed": True,
                })
            except Exception as e:
                print(f"[REST] 완결캔들 처리 오류: {e}")

        # 현재 진행 중인 캔들 처리 (마지막)
        if ohlcv:
            cur = ohlcv[-1]
            try:
                await _handle_candle({
                    "time":      cur[0] // 1000,
                    "open":      float(cur[1]),
                    "high":      float(cur[2]),
                    "low":       float(cur[3]),
                    "close":     float(cur[4]),
                    "volume":    float(cur[5]),
                    "is_closed": False,
                })
            except Exception as e:
                print(f"[REST] 현재캔들 처리 오류: {e}")

        await asyncio.sleep(POLL_INTERVAL_S)

# ─────────────────────────────────────────────
#  메인 Binance 피드 태스크
# ─────────────────────────────────────────────
async def _binance_feed():
    """WebSocket → REST 폴링 순서로 fallback하며 무한 재시도."""
    while True:
        print("[FEED] Binance WebSocket 연결 시도 중...")
        ws_ok = await _binance_ws_feed()
        if ws_ok:
            # WebSocket이 정상 종료됐으면 재연결 시도
            print("[FEED] WebSocket 연결 끊김 → 재연결 시도...")
            await asyncio.sleep(3)
        else:
            print("[FEED] 모든 WebSocket URL 실패 → REST 폴링으로 전환")
            await _binance_rest_feed()
            # REST 폴링은 무한루프이므로 여기까지 오지 않음 (서버 재시작 시에만)

# ─────────────────────────────────────────────
#  FastAPI 이벤트 & 엔드포인트
# ─────────────────────────────────────────────
@app.on_event("startup")
async def _startup():
    asyncio.create_task(_binance_feed())

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.add(ws)
    print(f"[CLIENT] 연결 (총 {len(connected_clients)}개)")

    # 신규 접속 시 저장된 분봉 중 가장 최근 시간대 로드 (재시작 후에도 이전 데이터 표시)
    now_ts = int(datetime.now().timestamp())
    hour_candles, period_start, period_end = _load_latest_hour_candles(now_ts)
    if hour_candles and period_start and period_end:
        try:
            is_past = (period_end <= now_ts)
            await ws.send_text(json.dumps({
                "type": "hour_history",
                "candles": hour_candles,
                "period_start": period_start,
                "period_end": period_end,
                "is_past": is_past,
            }, ensure_ascii=False))
            print(f"[CLIENT] hour_history 전송: {len(hour_candles)}개 (period {period_start}~{period_end})")
        except Exception as e:
            print(f"[CLIENT] hour_history 전송 실패: {e}")

    # 예측 히스토리 로드 (캔들과 동일 시간대)
    pred_ts = period_start if period_start else now_ts
    hour_predictions = _load_hour_predictions(pred_ts)
    if hour_predictions:
        try:
            await ws.send_text(json.dumps({
                "type": "prediction_history",
                "predictions": hour_predictions,
            }, ensure_ascii=False))
        except Exception as e:
            print(f"[CLIENT] prediction_history 전송 실패: {e}")

    try:
        while True:
            # 브라우저 메시지 수신 대기 (ping 등) — 연결 끊기면 WebSocketDisconnect 발생
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        connected_clients.discard(ws)
        print(f"[CLIENT] 연결 해제 (총 {len(connected_clients)}개)")

@app.post("/save_snapshot")
async def save_snapshot(request: Request):
    """프론트엔드에서 Base64 PNG를 전송받아 snapshots/ 에 저장."""
    body     = await request.json()
    raw      = body.get("data", "")
    filename = body.get("filename", "snapshot.png")

    if "," in raw:
        raw = raw.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(raw)
    except Exception as e:
        return JSONResponse({"error": f"base64 디코드 실패: {e}"}, status_code=400)

    safe = Path(filename).name
    if not safe.endswith(".png"):
        safe += ".png"
    (SNAPSHOT_DIR / safe).write_bytes(img_bytes)
    print(f"[SNAP] 저장: {safe} ({len(img_bytes):,} bytes)")
    return {"status": "ok", "filename": safe, "size": len(img_bytes)}

@app.get("/snapshots")
async def list_snapshots():
    files = sorted(SNAPSHOT_DIR.glob("*.png"),
                   key=lambda f: f.stat().st_mtime, reverse=True)
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
        "clients":        len(connected_clients),
        "candle_buffer":  len(candle_buffer),
        "model":          "TCN" if _model is not None else "linear_fallback",
        "last_close":     candle_buffer[-1]["close"] if candle_buffer else None,
    }


@app.get("/api/model_debug")
async def model_debug():
    """예측 검증용: 마지막 예측의 모델 유형, log_return 출력값, 가격 변환 결과."""
    return {
        "model_loaded": _model is not None,
        **_last_pred_debug,
    }


@app.get("/api/logs")
async def get_logs(tail: int = 1000):
    """서버 로그 파일 내용 (마지막 tail줄). txt 형식."""
    if not LOG_FILE.exists():
        return {"content": "", "path": str(LOG_FILE)}
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        content = "".join(lines[-tail:]) if len(lines) > tail else "".join(lines)
        return {"content": content, "path": str(LOG_FILE), "lines": len(lines)}
    except Exception as e:
        return {"content": f"[오류] {e}", "path": str(LOG_FILE)}


@app.get("/api/logs/download")
async def download_logs():
    """로그 파일 직접 다운로드."""
    if not LOG_FILE.exists():
        return JSONResponse({"error": "로그 파일 없음"}, status_code=404)
    return FileResponse(LOG_FILE, filename="server.txt", media_type="text/plain")

@app.get("/")
async def index():
    return FileResponse("index.html")

# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
