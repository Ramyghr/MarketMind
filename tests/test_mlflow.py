import mlflow

mlflow.set_experiment("marketmind")

with mlflow.start_run(run_name="hello-world"):
    mlflow.log_param("model", "test")
    mlflow.log_metric("fake_loss", 0.42)

print("MLflow working. Run: mlflow ui")