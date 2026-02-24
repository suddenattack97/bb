import ccxt
import pandas as pd
import time
from datetime import datetime
import os

def fetch_all_binance_1m(symbol="BTC/USDT", filename="BTC_all_1m.csv"):
    binance = ccxt.binance()
    
    # 1. 파일이 이미 있으면 마지막 날짜부터 이어받기 (이어쓰기 지원)
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename, index_col='timestamp', parse_dates=True)
        # 마지막 시간 + 1분(60000ms)
        start_time = int(existing_df.index[-1].timestamp() * 1000) + 60000 
        print(f"기존 데이터를 발견했습니다. {existing_df.index[-1]} 이후부터 이어받습니다.")
    else:
        # 바이낸스 BTC/USDT 상장일 (2017년 8월 17일경) timestamp
        start_time = 1502942400000 
        print("처음부터 모든 데이터를 수집합니다 (2017-08-17 ~).")
        
    end_time = binance.milliseconds()
    current_since = start_time
    
    # 2. 반복문으로 1000개씩 크롤링 (API 제한 주의)
    while current_since < end_time:
        try:
            ohlcv = binance.fetch_ohlcv(symbol, timeframe='1m', since=current_since, limit=1000)
            if not ohlcv:
                break
                
            # 데이터프레임 변환
            df_chunk = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'], unit='ms')
            df_chunk.set_index('timestamp', inplace=True)
            
            # 파일에 이어쓰기 (처음엔 헤더 포함, 이후엔 헤더 제외)
            write_header = not os.path.exists(filename)
            df_chunk.to_csv(filename, mode='a', header=write_header)
            
            current_since = ohlcv[-1][0] + 60000 # 다음 호출을 위한 타임스탬프 업데이트
            collected_date = datetime.fromtimestamp(ohlcv[-1][0] / 1000.0)
            
            print(f"[{collected_date.strftime('%Y-%m-%d %H:%M:%S')}] 수집 완료 (누적 저장 중...)")
            time.sleep(0.5) # 바이낸스 API 밴 방지 (초당 2~3회 호출)
            
        except Exception as e:
            print(f"API 오류 발생: {e}, 10초 후 재시도...")
            time.sleep(10)
            
    print("✅ 모든 과거 1분봉 데이터 수집이 완료되었습니다!")

if __name__ == "__main__":
    fetch_all_binance_1m()
