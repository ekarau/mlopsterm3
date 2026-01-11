import sys
import os
import joblib
import warnings
import shutil
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder

# --- Path Setup for Docker Environment ---
# We add /opt/airflow to sys.path so Python can find custom modules inside the container
sys.path.append('/opt/airflow')

try:
    from src.ingest import load_data
    from src.preprocess import clean_data, balance_data
    from src.features import apply_feature_cross, apply_hashing
except ImportError:
    # Fallback for local testing outside Docker
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.ingest import load_data
    from src.preprocess import clean_data, balance_data
    from src.features import apply_feature_cross, apply_hashing

warnings.filterwarnings("ignore")

# --- Constants ---
# Defined absolute paths for Docker environment
CHECKPOINT_DIR = '/opt/airflow/data/models'
DATA_PATH = '/opt/airflow/data/raw/Course_Completion_Prediction.csv'
BACKUP_DATA_PATH = '/opt/airflow/data/interim/3_features.csv'

class MLEngineerPipeline:
    """
    Class responsible for running ML experiments, logging to MLflow,
    and saving model artifacts.
    """
    def __init__(self, processed_dataframe, experiment_name="Course_Completion_MLOps"):
        self.data = processed_dataframe
        self.experiment_name = experiment_name
        self.results = []

        # Initialize MLflow experiment
        mlflow.set_experiment(self.experiment_name)

        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    def run_classification_experiments(self):
        """
        Runs standard classification models (RandomForest, XGBoost).
        """
        target_col = 'target' if 'target' in self.data.columns else 'Completed'

        if target_col not in self.data.columns:
            print(f"Target column '{target_col}' not found. Available columns: {self.data.columns}")
            raise ValueError(f"Target column '{target_col}' missing.")

        # Drop target and auxiliary columns
        drop_cols = [target_col, 'Progress_Percentage']
        cols_to_drop = [c for c in drop_cols if c in self.data.columns]

        X = self.data.drop(columns=cols_to_drop)
        y = self.data[target_col]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define models to compare
        models = {
            "RandomForest_Bagging": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            "XGBoost_Boosting": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, eval_metric="logloss", random_state=42)
        }

        # Train and Evaluate
        for name, model in models.items():
            with mlflow.start_run(run_name=name):
                print(f"Training {name}...")
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds)

                # Log metrics and params
                mlflow.log_param("model_type", name)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score", f1)

                mlflow.sklearn.log_model(model, "model")
                
                # Save model locally
                joblib.dump(model, f"{CHECKPOINT_DIR}/{name}.pkl")

                self.results.append({
                    "Model": name,
                    "Task": "Classification",
                    "Accuracy": acc,
                    "F1": f1,
                    "RMSE": None
                })
                print(f"  {name} -> Accuracy: {acc:.4f}")

    def run_reframing_experiment(self):
        """
        Experimental approach: Treat as Regression (predict %) then convert to Classification.
        """
        if 'Progress_Percentage' not in self.data.columns:
            print("Skipping reframing experiment: 'Progress_Percentage' column missing.")
            return

        target_col = 'target' if 'target' in self.data.columns else 'Completed'

        cols_to_drop = [target_col, 'Progress_Percentage']
        real_drop = [c for c in cols_to_drop if c in self.data.columns]

        X = self.data.drop(columns=real_drop)
        y_reg = self.data['Progress_Percentage']
        y_class_true = self.data[target_col]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        _, _, _, y_test_class = train_test_split(X, y_class_true, test_size=0.2, random_state=42)

        model_name = "XGBoost_Reframed_Regressor"
        model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)

        with mlflow.start_run(run_name=model_name):
            print(f"Training {model_name}...")
            model.fit(X_train, y_train)
            preds_percent = model.predict(X_test)

            # Convert Regression output to Classification (Threshold: 50%)
            preds_class = [1 if p >= 50.0 else 0 for p in preds_percent]

            rmse = np.sqrt(mean_squared_error(y_test, preds_percent))
            acc = accuracy_score(y_test_class, preds_class)

            mlflow.log_param("model_type", model_name)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("derived_accuracy", acc)

            mlflow.sklearn.log_model(model, "model")
            joblib.dump(model, f"{CHECKPOINT_DIR}/{model_name}.pkl")

            self.results.append({
                "Model": model_name,
                "Task": "Reframing (Reg->Clf)",
                "Accuracy": acc,
                "F1": None,
                "RMSE": rmse
            })
            print(f"  {model_name} -> Derived Accuracy: {acc:.4f}")

    def get_results_table(self):
        return pd.DataFrame(self.results).sort_values(by="Accuracy", ascending=False)

# --- MAIN EXECUTION FUNCTION ---
# This function is what Airflow imports and runs.
def main():
    print("🚀 Training process started inside Airflow...")
    
    try:
        # 1. Load Data
        # Try loading raw data first, otherwise fallback to interim data
        if os.path.exists(DATA_PATH):
            print(f"Loading data from {DATA_PATH}")
            raw_df = load_data(DATA_PATH)
        elif os.path.exists(BACKUP_DATA_PATH):
            print(f"Raw data not found. Loading interim data from {BACKUP_DATA_PATH}")
            raw_df = pd.read_csv(BACKUP_DATA_PATH)
        else:
            raise FileNotFoundError(f"Data not found at {DATA_PATH} or {BACKUP_DATA_PATH}")

        # 2. Pipeline Steps (Preprocessing & Feature Engineering)
        # Note: Even if Airflow did these steps, we re-run them here to ensure
        # the training script is self-contained and consistent.
        print("Preprocessing data...")
        clean_df = clean_data(raw_df)
        balanced_df = balance_data(clean_df)
        df_crossed = apply_feature_cross(balanced_df)

        if 'Student_ID' in df_crossed.columns:
            final_df = apply_hashing(df_crossed, 'Student_ID', n_features=50)
        else:
            final_df = df_crossed

        # Label Encoding for Object columns
        object_cols = final_df.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in object_cols:
            if col != 'target' and 'hashed' not in col:
                final_df[col] = le.fit_transform(final_df[col].astype(str))

        print(f"Data Ready for Training. Shape: {final_df.shape}")

        # 3. Run Experiments
        pipeline = MLEngineerPipeline(final_df)
        pipeline.run_classification_experiments()
        pipeline.run_reframing_experiment()

        # 4. Report Results
        print("\n--- EXPERIMENT RESULTS REPORT ---")
        print(pipeline.get_results_table())
        print(f"\nModels saved to '{CHECKPOINT_DIR}' directory.")
        
        # 5. Save the Best Model for API Usage
        # We copy the standard XGBoost model to 'model.pkl' so the API can find it easily.
        best_model_source = f"{CHECKPOINT_DIR}/XGBoost_Boosting.pkl"
        final_model_dest = f"{CHECKPOINT_DIR}/model.pkl"
        
        if os.path.exists(best_model_source):
            shutil.copy(best_model_source, final_model_dest)
            print(f"✅ Best model copied to {final_model_dest} for API usage.")
        else:
            print("⚠️ Warning: Could not find XGBoost model to set as default.")

    except Exception as e:
        print(f"❌ Critical Error in Training: {e}")
        # Re-raise the exception so Airflow marks the task as FAILED
        raise e

# This block allows running the script manually from terminal
if __name__ == "__main__":
    main()