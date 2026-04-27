import pandas as pd
from pathlib import Path

for f in Path('data/raw').glob('*.parquet'):
    df = pd.read_parquet(f)
    nulls = df.isna().sum().sum()
    print(f"{f.name}: {df.shape[0]} rows | {df.index[0].date()} -> {df.index[-1].date()} | {nulls} nulls")
