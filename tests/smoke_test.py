import sys
sys.path.insert(0, 'src')
from data.dataset import MarketWindowDataset, load_split

paths = [
    'data/raw/BTCUSDT_1h.parquet',
    'data/raw/ETHUSDT_1h.parquet',
    'data/raw/SPY_1d.parquet',
    'data/raw/QQQ_1d.parquet',
]

ds = load_split(paths, split='train')
print('train size:', len(ds))
x = ds[0]
print('shape:', x.shape)
print('mean (should be ~0):', round(float(x.mean()), 4))

for split in ['val', 'test']:
    d = load_split(paths, split=split)
    print(f'{split} size:', len(d))
