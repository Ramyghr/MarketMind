import sys
sys.path.insert(0, 'src')
import torch
from ssl_model.encoder import MarketEncoder
from ssl_model.losses import nt_xent_loss

enc = MarketEncoder()
x1 = torch.randn(32, 60, 5)
x2 = torch.randn(32, 60, 5)
z1, z2 = enc(x1), enc(x2)
print('z1 shape:', z1.shape)  # (32, 128)

loss = nt_xent_loss(z1, z2)
print('loss:', loss.item())
assert z1.shape == (32, 128)
assert not torch.isnan(loss)
assert not torch.isinf(loss)
print('Day 9: OK')