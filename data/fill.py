import pandas as pd
from pathlib import Path

for f in Path('data/raw').glob('*.parquet'):
    df = pd.read_parquet(f)
    df = df.ffill(limit=3)
    pct = df.isna().sum() / len(df)
    assert (pct < 0.01).all(), f'Too many nulls: {f.name}'
    df.to_parquet(f)
    print(f'{f.name}: ffill done')