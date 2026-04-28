import torch
import torch.nn as nn

class MarketEncoder(nn.Module):
    def __init__(self, n_features=5, d_model=64, nhead=4, n_layers=2, out_dim=128):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead,
            dim_feedforward=256, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, out_dim))

    def forward(self, x):
        x = self.input_proj(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        return self.proj(x[:, 0, :])