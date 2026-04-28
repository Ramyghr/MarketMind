import sys
sys.path.insert(0, 'src')
import torch
import mlflow
import mlflow.pytorch
from ssl_model.encoder import MarketEncoder

mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('ssl-pretraining')

checkpoint = torch.load('checkpoints/ssl_encoder_30ep.pt', map_location='cpu')
model = MarketEncoder()
model.load_state_dict(checkpoint['model_state'])

with mlflow.start_run(run_name='ssl-multi-asset-v1'):
    mlflow.log_params({
        'lr': 3e-4,
        'batch_size': 256,
        'epochs': 30,
        'window': 60,
        'embed_dim': 128,
        'temperature': 0.07,
        'assets': 'BTC+ETH+SPY+QQQ',
        'date_range': '2018-2024'
    })
    mlflow.log_metric('final_loss', checkpoint['final_loss'])
    mlflow.log_artifact('checkpoints/ssl_encoder_30ep.pt')
    mlflow.log_artifact('notebooks/figures/tsne_btc_ssl.png')
    mlflow.pytorch.log_model(model, 'ssl_encoder',
        registered_model_name='MarketMindEncoder')
    print('run logged')