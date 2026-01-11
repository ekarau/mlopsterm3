import os
from prefect import flow, task
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

@task
def train_and_log():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("prefect-demo")

    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200).fit(X, y)

    with mlflow.start_run(run_name="prefect-logreg"):
        mlflow.log_param("model_type", "logreg")
        mlflow.log_metric("train_accuracy", model.score(X, y))
        mlflow.sklearn.log_model(
            model, "model",
            registered_model_name="course-completion-model"
        )

@flow(name="train-flow")
def train_flow():
    train_and_log()


if __name__ == "__main__":
    train_flow()
    
