import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# MLflow server adresi (compose içinden geliyor)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

# Experiment
mlflow.set_experiment("demo-registry")

# Basit demo modeli
X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200)
model.fit(X, y)

with mlflow.start_run(run_name="logreg-demo"):
    mlflow.log_param("model_type", "logistic_regression")
    mlflow.log_metric("train_accuracy", model.score(X, y))

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="course-completion-model"
    )

print("Run logged & model registered")
