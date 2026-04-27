import yfinance as yf
for ticker in ['SPY', 'QQQ']:
    df = yf.download(ticker, start='2018-01-01', end='2025-01-01')
    df.columns = df.columns.get_level_values(0).str.lower()
    df[['open','high','low','close','volume']].to_parquet(f'data/raw/{ticker}_1d.parquet')
