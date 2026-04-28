import sys
sys.path.insert(0, 'src')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# load embeddings
embeddings = np.load('data/embeddings/test_emb.npy')
print('embeddings:', embeddings.shape)

# load BTC rolling return as regime proxy
btc = pd.read_parquet('data/raw/BTCUSDT_1h.parquet')
btc_test = btc[btc.index >= '2024-01-01']
returns = btc_test['close'].pct_change(30*24).fillna(0).values
returns = np.tile(returns, (len(embeddings) // len(returns) + 1))[:len(embeddings)]

# run t-SNE
print('running t-SNE...')
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
reduced = tsne.fit_transform(embeddings)

# plot
import os
os.makedirs('notebooks/figures', exist_ok=True)
plt.figure(figsize=(10, 7))
scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=returns, cmap='RdYlGn', s=2, alpha=0.6, vmin=-0.3, vmax=0.3)
plt.colorbar(scatter, label='30-day rolling return (BTC)')
plt.title('t-SNE of SSL Market Embeddings (2024 test set)')
plt.savefig('notebooks/figures/tsne_btc_ssl.png', dpi=150, bbox_inches='tight')
print('saved to notebooks/figures/tsne_btc_ssl.png')
plt.show()