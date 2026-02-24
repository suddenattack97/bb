# 비트코인 1분봉 실시간 예측 시스템 설계서 v2 (적중률 검증 UI)

## 1. 시스템 핵심 목표 및 특징
이번 설계는 모델의 **과거 예측(점선)이 실제 현재가(실선/캔들)와 얼마나 일치하는지 시각적으로 검증**하기 위해 30분 단위 고정 타임라인과 스냅샷 아카이빙 기능을 제공하는 것이 핵심입니다.

- **고정 타임프레임 렌더링 (30분 단위):** 차트의 X축은 항상 `00:00 ~ 00:30`, `00:30 ~ 01:00` 등 30분 단위로 강제 고정됩니다. 현재 시간이 00:14분이어도 차트는 00:30분까지의 빈 공간을 우측에 렌더링합니다.
- **예측 궤적 영구 보존:** 00:10분에 모델이 예측한 미래 5분 궤적은 시간이 지나 00:15분이 되어 실제 캔들이 그려져도 지워지지 않습니다. 차트 위에 누적된 예측 선(과거의 예측값)들과 실제 캔들을 동시에 띄워 **'예측이 맞았는지 틀렸는지' 육안으로 비교**합니다.
- **자동 Y축 스케일링:** 30분 동안의 실제 최고/최저가와 모델이 예측했던 최고/최저가 모두를 포함할 수 있도록 Y축 스케일이 실시간으로 자동 조절됩니다.
- **스냅샷 자동 저장:** 정각 00:30분, 01:00분이 되면 30분간 누적된 '실제 캔들 + 예측 궤적'을 HTML5 Canvas에서 이미지(`.png`)로 스냅샷을 찍어 서버에 저장하고, 차트는 다음 30분 주기로 초기화됩니다.

---

## 2. 화면 레이아웃 구성

```text
+-------------------------------------------------------------+
| 🎯 BTC/USDT 30분 단위 적중률 검증 차트 (고정 타임라인)              |
| 현재 구간: 00:00 ~ 00:30 | 현재 시간: 00:14 | 웹소켓 상태: 🟢     |
+-------------------------------------------------------------+
|                                                             |
|   [$64,250] - Y축(자동 스케일링)                              |
|   [      ]                                                  |
|   [      ]                       + (10분에 예측했던 15분 목표) |
|   [      ]                      /                           |
|   [      ]                     /                 (빈 공간)   |
|   [      ]       + (과거 예측선) -+ (실제 14분 캔들)            |
|   [      ]      /             /                             |
|   [      ]     /             /                              |
|   [$64,150] --+ (실제 10분) -+                               |
|                                                             |
+-------------------------------------------------------------+
| [00:00]      [00:10]      [00:20]      [00:30] (X축 고정)    |
+-------------------------------------------------------------+
| 📷 지난 스냅샷 보기: [09:30~10:00.png] [10:00~10:30.png]        |
+-------------------------------------------------------------+
```

---

## 3. 프론트엔드 핵심 코드 (Vanilla JS + Lightweight Charts)

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>BTC 1분봉 적중률 검증 차트</title>
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        body { font-family: 'Malgun Gothic', sans-serif; background: #131722; color: white; margin: 0; padding: 20px; }
        #chart-container { width: 100%; height: 600px; border: 1px solid #2B2B43; }
        .info-panel { margin-bottom: 10px; padding: 10px; background: #1E222D; border-radius: 5px; display: flex; justify-content: space-between; }
    </style>
</head>
<body>
    <div class="info-panel">
        <div>
            <h2>🎯 BTC 적중률 검증 차트</h2>
            <span id="period">현재 구간: 계산 중...</span> | 
            <span id="time">현재 시간: 00:00</span>
        </div>
        <div>
            <button id="btn-snapshot" style="padding: 10px; background: #2962FF; color: white; border: none; cursor: pointer;">수동 스냅샷 캡처</button>
        </div>
    </div>
    <div id="chart-container"></div>

    <script>
        // 1. 차트 기본 설정 (오른쪽 빈 공간을 강제로 확보하고 Y축 자동 스케일 활성화)
        const chart = LightweightCharts.createChart(document.getElementById('chart-container'), {
            layout: { backgroundColor: '#131722', textColor: '#d1d4dc' },
            grid: { vertLines: { color: '#2B2B43' }, horzLines: { color: '#2B2B43' } },
            timeScale: { 
                timeVisible: true, 
                secondsVisible: false,
                rightOffset: 0, // 고정 타임라인이므로 오프셋 사용 안함
                fixLeftEdge: true, 
                fixRightEdge: true 
            },
            rightPriceScale: { 
                autoScale: true, // Y축 자동 스케일링 (예측가가 화면 밖으로 나가지 않음)
                alignLabels: true
            }
        });

        const candleSeries = chart.addCandlestickSeries({
            upColor: '#26a69a', downColor: '#ef5350', borderVisible: false, wickUpColor: '#26a69a', wickDownColor: '#ef5350'
        });

        // 과거 예측선들을 저장할 배열 (1분마다 5분짜리 예측선 시리즈가 새로 생성되어 영구 보존됨)
        let predictionSeriesList = [];

        // 30분 단위 고정 X축 세팅 함수
        function setFixedTimeline() {
            const now = new Date();
            const minutes = now.getMinutes();
            // 현재 시간이 0~29분이면 시작=00분, 끝=30분 / 30~59분이면 시작=30분, 끝=60분(다음시간 정각)
            const startMin = minutes < 30 ? 0 : 30;
            const endMin = minutes < 30 ? 30 : 60;

            const startTime = new Date(now.getFullYear(), now.getMonth(), now.getDate(), now.getHours(), startMin, 0);
            const endTime = new Date(now.getFullYear(), now.getMonth(), now.getDate(), now.getHours(), endMin, 0);

            document.getElementById('period').innerText = `현재 구간: ${startTime.toTimeString().substring(0,5)} ~ ${endTime.toTimeString().substring(0,5)}`;

            // X축 강제 고정을 위해 시작시간과 끝시간에 투명한(보이지 않는) 더미 데이터를 삽입
            candleSeries.setData([
                { time: startTime.getTime() / 1000, open: null, high: null, low: null, close: null },
                { time: endTime.getTime() / 1000, open: null, high: null, low: null, close: null }
            ]);

            return endTime;
        }

        let currentEndTime = setFixedTimeline();

        // 자동 스냅샷 캡처 및 초기화 함수
        function takeSnapshotAndReset() {
            // 차트 캔버스를 Base64 이미지로 변환
            const canvas = document.querySelector('.tv-lightweight-charts canvas');
            const dataURL = canvas.toDataURL('image/png');

            // (서버로 전송하는 로직 추가 필요)
            console.log("📸 스냅샷 캡처 완료!");

            // 차트 초기화 (다음 30분을 위해)
            candleSeries.setData([]);
            predictionSeriesList.forEach(series => chart.removeSeries(series));
            predictionSeriesList = [];
            currentEndTime = setFixedTimeline();
        }

        // 수동 스냅샷 버튼 이벤트
        document.getElementById('btn-snapshot').addEventListener('click', takeSnapshotAndReset);

        // 파이썬 웹소켓 연결
        const ws = new WebSocket('ws://localhost:8000/ws'); 

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            const now = new Date();
            document.getElementById('time').innerText = `현재 시간: ${now.toTimeString().substring(0,5)}`;

            // 정해진 30분이 지나면 스냅샷 찍고 초기화
            if (now >= currentEndTime) {
                takeSnapshotAndReset();
            }

            // 실제 1분봉 데이터 렌더링
            if (data.type === 'real_time_candle') {
                candleSeries.update(data.candle); 
            }

            // AI가 예측한 미래 5개의 좌표 렌더링 (지우지 않고 매번 새로운 시리즈로 오버레이)
            if (data.type === 'prediction' && data.predictions) {
                const newPredSeries = chart.addLineSeries({
                    color: `rgba(41, 98, 255, 0.4)`, // 겹쳐 보일 수 있도록 반투명 파란색 처리
                    lineWidth: 2, 
                    lineStyle: 2, 
                    crosshairMarkerVisible: false, 
                    lastValueVisible: false
                });
                newPredSeries.setData(data.predictions);
                predictionSeriesList.push(newPredSeries); // 배열에 저장하여 유지
            }
        };
    </script>
</body>
</html>
```

## 4. 운영 시나리오 (적중률 시각적 검증 프로세스)
1. **00:00 분 (구간 시작):** 차트의 X축이 `00:00 ~ 00:30`으로 고정됩니다. 오른쪽 절반 이상이 빈 공간으로 열려 있습니다.
2. **00:01 분:** 첫 번째 캔들이 찍히고, AI가 모델을 돌려 00:02~00:06까지의 예상 가격을 반투명한 파란 점선으로 그립니다.
3. **00:05 분:** 실제 캔들이 계속 그려지면서 과거 00:01분에 그렸던 파란 점선을 덮거나 통과합니다. 사용자는 **"아까 1분에 예측했던 선(점선)과 현재 캔들(실선)이 정확히 겹치네/빗나갔네"**를 눈으로 바로 확인할 수 있습니다.
4. **00:30 분 (구간 종료):** 차트 화면의 캔버스 렌더링이 `dataURL` 기반의 PNG 이미지로 캡처됩니다. 이 이미지는 파이썬 백엔드로 전송되어 `/snapshots/` 폴더에 `2026-02-24_0000_0030.png` 형태로 영구 저장됩니다. 
5. 차트 내부의 모든 캔들과 수십 개의 반투명 점선 궤적들이 초기화되고, X축이 `00:30 ~ 01:00`으로 변경되며 새로운 사이클이 시작됩니다.
