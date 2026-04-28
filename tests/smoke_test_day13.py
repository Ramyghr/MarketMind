import sys
sys.path.insert(0, 'src')
import numpy as np
from rl.trading_env import TradingEnv
from gymnasium.utils.env_checker import check_env

# load real embeddings and prices
embeddings = np.load('data/embeddings/test_emb.npy').astype(np.float32)

import pandas as pd
btc = pd.read_parquet('data/raw/BTCUSDT_1h.parquet')
btc_test = btc[btc.index >= '2024-01-01']
prices = btc_test['close'].values
embeddings = embeddings[:len(prices)]

env = TradingEnv(embeddings, prices)

# check env
check_env(env)
print('check_env: OK')

# buy & hold
obs, _ = env.reset()
while True:
    obs, r, done, _, _ = env.step(1)
    if done: break
bh_portfolio = env.portfolio
print(f'buy&hold portfolio: {bh_portfolio:.4f}')

# random agent
obs, _ = env.reset()
while True:
    obs, r, done, _, _ = env.step(env.action_space.sample())
    if done: break
rand_portfolio = env.portfolio
print(f'random portfolio: {rand_portfolio:.4f}')

assert bh_portfolio > rand_portfolio, 'buy&hold should beat random on 2024 BTC bull year'
print('sanity check: OK')