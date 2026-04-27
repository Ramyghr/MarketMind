import requests, pandas as pd, time

def fetch_binance(symbol, interval, start, end):
    url = 'https://api.binance.com/api/v3/klines'
    all_data, start_ts = [], int(pd.Timestamp(start).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end).timestamp() * 1000)
    while start_ts < end_ts:
        r = requests.get(url, params={'symbol': symbol,
            'interval': interval, 'startTime': start_ts, 'limit': 1000}).json()
        if not r: break
        all_data.extend(r)
        start_ts = r[-1][0] + 1
        time.sleep(0.1)
    cols = ['ts','open','high','low','close','volume',
            'close_time','qav','trades','tbav','tqav','ignore']
    df = pd.DataFrame(all_data, columns=cols)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    return df.set_index('ts')[['open','high','low','close','volume']].astype(float)

for sym in ['BTCUSDT', 'ETHUSDT']:
    df = fetch_binance(sym, '1h', '2018-01-01', '2025-01-01')
    df = df[df.index < '2025-01-01']
    df.to_parquet(f'data/raw/{sym}_1h.parquet')
    print(f'{sym}: {len(df)} rows')