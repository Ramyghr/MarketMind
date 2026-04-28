import sys
sys.path.insert(0, 'src')
import torch
import numpy as np
from torch.utils.data import DataLoader
from ssl_model.encoder import MarketEncoder
from data.dataset import load_split

paths = [
    'data/raw/BTCUSDT_1h.parquet',
    'data/raw/ETHUSDT_1h.parquet',
    'data/raw/SPY_1d.parquet',
    'data/raw/QQQ_1d.parquet',
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load('checkpoints/ssl_encoder_30ep.pt', map_location=device)
model = MarketEncoder()
model.load_state_dict(checkpoint['model_state'])
model.eval().to(device)

test_ds = load_split(paths, split='test')
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

embeddings = []
with torch.no_grad():
    for batch in test_loader:
        x = batch.float().to(device)
        z = model(x)
        embeddings.append(z.cpu().numpy())

embeddings = np.concatenate(embeddings)
print('embeddings shape:', embeddings.shape)

import os
os.makedirs('data/embeddings', exist_ok=True)
np.save('data/embeddings/test_emb.npy', embeddings)
print('saved to data/embeddings/test_emb.npy')