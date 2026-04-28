import sys
sys.path.insert(0, 'src')
import torch
import mlflow.pytorch

mlflow.set_tracking_uri('sqlite:///mlflow.db')
model = mlflow.pytorch.load_model('models:/MarketMindEncoder/1')
model.eval()
z = model(torch.randn(4, 60, 5).float())
print('round-trip shape:', z.shape)  # (4, 128)