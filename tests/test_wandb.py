import wandb

run = wandb.init(project="marketmind", name="hello-world")
for i in range(10):
    wandb.log({"fake_loss": 1.0 / (i + 1), "epoch": i})
wandb.finish()
print("W&B working.")
