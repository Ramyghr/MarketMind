import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class MarketWindowDataset(Dataset):
    def __init__(self, parquet_paths, window=60, stride=1, augment_fn=None):
        self.windows = []
        for path in parquet_paths:
            df = pd.read_parquet(path)[['open','high','low','close','volume']]
            arr = df.values.astype(np.float32)
            for i in range(0, len(arr) - window, stride):
                w = arr[i:i+window]
                self.windows.append(self._zscore(w))
        self.windows = np.stack(self.windows)
        self.augment_fn = augment_fn

    def _zscore(self, w):
        return (w - w.mean(0)) / (w.std(0) + 1e-8)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx]
        if self.augment_fn:
            return torch.tensor(self.augment_fn(x.copy())), torch.tensor(self.augment_fn(x.copy()))
        return torch.tensor(x)


def load_split(parquet_paths, split='train', window=60, stride=1, augment_fn=None):
    splits = {
        'train': (None, '2023-01-01'),
        'val':   ('2023-01-01', '2024-01-01'),
        'test':  ('2024-01-01', None),
    }
    start, end = splits[split]
    filtered = []
    for path in parquet_paths:
        df = pd.read_parquet(path)
        if start: df = df[df.index >= start]
        if end:   df = df[df.index < end]
        tmp = f'/tmp/split_{split}_{path.split("/")[-1]}'
        df.to_parquet(tmp)
        filtered.append(tmp)
    return MarketWindowDataset(filtered, window=window, stride=stride, augment_fn=augment_fn)