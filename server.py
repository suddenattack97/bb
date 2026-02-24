"""
BTC/USDT 1분봉 실시간 예측 웹서버

Binance 연결 전략 (순서대로 시도):
  1) websockets 라이브러리  - stream.binance.com:9443  (표준)
  2) websockets 라이브러리  - stream.binance.com:443   (방화벽 우회)
  3) websockets 라이브러리  - data-stream.binance.vision (CDN)
  4) ccxt REST 폴링 fallback - 5초 간격

실행: uvicorn server:app --host 0.0.0.0 --port 8000
"""
import asyncio
import base64
import json
import ssl
import sys
from pathlib import Path
from typing import Optional

# Windows CP949 콘솔에서 이모지 출력 오류 방지
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

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
SNAPSHOT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="BTC 1분봉 예측 서버")

# ─────────────────────────────────────────────
#  전역 상태
# ─────────────────────────────────────────────
connected_clients: set[WebSocket] = set()
candle_buffer: list[dict]         = []   # 완결 캔들 저장 (최대 60개)
_prev_candle_ts: int              = 0    # 마지막 완결 캔들 타임스탬프 (중복 예측 방지)

# ─────────────────────────────────────────────
#  모델 로드 (선택적)
# ─────────────────────────────────────────────
_model       = None
_scaler_mean: Optional[np.ndarray] = None
_scaler_std:  Optional[np.ndarray] = None

def _load_model():
    global _model, _scaler_mean, _scaler_std
    if not TORCH_AVAILABLE:
        print("[MODEL] PyTorch 없음 → 선형 추정 fallback 사용")
        return
    try:
        import torch
        from model import TCNForecaster
        mp, sp = Path("tcn_base_model.pth"), Path("scaler.npy")
        if mp.exists() and sp.exists():
            m = TCNForecaster(num_features=4, output_steps=PRED_STEPS)
            m.load_state_dict(torch.load(str(mp), map_location="cpu"))
            m.eval()
            _model = m
            sc = np.load(str(sp), allow_pickle=True).item()
            _scaler_mean, _scaler_std = sc["mean"], sc["std"]
            print("[MODEL] TCN 모델 로드 완료")
        else:
            print("[MODEL] 모델 파일 없음 → 선형 추정 fallback 사용")
    except Exception as e:
        print(f"[MODEL] 로드 실패: {e} → 선형 추정 fallback 사용")

_load_model()

# ─────────────────────────────────────────────
#  예측 로직
# ─────────────────────────────────────────────
def _run_prediction(buf: list[dict]) -> Optional[list[dict]]:
    if len(buf) < 5:
        return None
    prices  = [c["close"]  for c in buf]
    volumes = [c["volume"] for c in buf]
    last_ts = buf[-1]["time"]

    if _model is not None:
        try:
            import torch
            closes = np.array(prices, dtype=np.float64)
            log_ret = np.diff(np.log(closes), prepend=np.log(closes[0]))
            delta    = np.diff(closes, prepend=closes[0])
            gain     = np.where(delta > 0, delta, 0.0)
            loss     = np.where(delta < 0, -delta, 0.0)
            avg_gain = np.convolve(gain, np.ones(14) / 14, mode="same")
            avg_loss = np.convolve(loss, np.ones(14) / 14, mode="same")
            rs       = np.where(avg_loss == 0, 100.0, avg_gain / (avg_loss + 1e-10))
            rsi      = 100 - (100 / (1 + rs))
            vols     = np.array(volumes, dtype=np.float64)
            feats    = np.stack([log_ret, rsi, closes, vols], axis=1).astype(np.float32)
            if _scaler_mean is not None:
                feats = (feats - _scaler_mean) / (_scaler_std + 1e-8)
            x = torch.tensor(feats[-MAX_CANDLE_BUFFER:]).unsqueeze(0)
            with torch.no_grad():
                pred_lr = _model(x).squeeze().numpy()
            last_c, pred_prices = closes[-1], []
            for lr in pred_lr:
                last_c = float(last_c) * float(np.exp(lr))
                pred_prices.append(last_c)
        except Exception as e:
            print(f"[PRED] 모델 추론 오류: {e} → fallback")
            pred_prices = predict_linear(prices, PRED_STEPS)
    else:
        pred_prices = predict_linear(prices, PRED_STEPS)

    return [
        {"time": last_ts + (i + 1) * 60, "value": round(float(p), 2)}
        for i, p in enumerate(pred_prices)
    ]

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
    """캔들 수신 시 실시간 업데이트 + 마감 시 예측 브로드캐스트."""
    global _prev_candle_ts

    # ① 항상 현재 캔들 전송
    await _broadcast({
        "type": "real_time_candle",
        "candle": {
            "time":  candle["time"],
            "open":  candle["open"],
            "high":  candle["high"],
            "low":   candle["low"],
            "close": candle["close"],
        },
    })

    # ② 캔들 마감 시에만 예측 실행 (중복 방지)
    if candle["is_closed"] and candle["time"] != _prev_candle_ts:
        _prev_candle_ts = candle["time"]
        candle_buffer.append(candle)
        if len(candle_buffer) > MAX_CANDLE_BUFFER:
            candle_buffer.pop(0)
        predictions = _run_prediction(candle_buffer)
        if predictions:
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

    # 신규 접속 시 최근 캔들 히스토리 즉시 전송 (차트 즉시 그리기)
    if candle_buffer:
        for c in candle_buffer[-30:]:
            try:
                await ws.send_text(json.dumps({
                    "type": "real_time_candle",
                    "candle": {
                        "time":  c["time"],
                        "open":  c["open"],
                        "high":  c["high"],
                        "low":   c["low"],
                        "close": c["close"],
                    },
                }))
            except Exception:
                break

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

@app.get("/")
async def index():
    return FileResponse("index.html")

# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
