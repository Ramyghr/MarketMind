import sys
sys.path.insert(0, 'src')
import torch
import wandb
from torch.utils.data import DataLoader
from ssl_model.encoder import MarketEncoder
from ssl_model.losses import nt_xent_loss
from ssl_model.augmentations import augment
from data.dataset import load_split

config = {
    'lr': 3e-4,
    'batch_size': 256,
    'epochs': 5,
    'temperature': 0.07,
    'embed_dim': 128,
    'data': '2018-2024'
}

wandb.init(project='marketmind', name='ssl-btc-v1', config=config)

paths = [
    'data/raw/BTCUSDT_1h.parquet',
    'data/raw/ETHUSDT_1h.parquet',
    'data/raw/SPY_1d.parquet',
    'data/raw/QQQ_1d.parquet',
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

print('loading dataset...')
train_ds = load_split(paths, split='train', augment_fn=augment)
print(f'dataset loaded: {len(train_ds)} windows')
train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=0)

model = MarketEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

import os
os.makedirs('checkpoints', exist_ok=True)

for epoch in range(config['epochs']):
    model.train()
    total_loss = 0
    for x1, x2 in train_loader:
        x1, x2 = x1.float().to(device), x2.float().to(device)
        z1, z2 = model(x1), model(x2)
        loss = nt_xent_loss(z1, z2, temperature=config['temperature'])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        wandb.log({'batch_loss': loss.item()})
    epoch_loss = total_loss / len(train_loader)
    scheduler.step()
    wandb.log({'epoch_loss': epoch_loss, 'epoch': epoch})
    print(f'epoch {epoch+1} loss: {epoch_loss:.4f}')

torch.save({
    'epoch': config['epochs'],
    'model_state': model.state_dict(),
    'config': config,
    'final_loss': epoch_loss
}, 'checkpoints/ssl_encoder_5ep.pt')
print('checkpoint saved')