from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import sys
import os

# Add /opt/airflow to the python path so the src module can be found inside Docker
sys.path.append('/opt/airflow')

# Import your custom modules
from src.ingest import load_data
from src.validate import validate_input_data
from src.preprocess import clean_data, split_data, balance_data
from src.features import apply_feature_cross, apply_hashing
# --- NEW: Import the training logic ---
from src.train_model import main as train_model_main

# File Paths (Using interim steps to create a visual lineage in Airflow)
DATA_DIR = '/opt/airflow/data'
RAW_PATH = f'{DATA_DIR}/raw/Course_Completion_Prediction.csv'
STAGE_1_VALIDATED = f'{DATA_DIR}/interim/1_validated.csv'
STAGE_2_CLEANED = f'{DATA_DIR}/interim/2_cleaned.csv'
STAGE_3_FEATURES = f'{DATA_DIR}/interim/3_features.csv'
PROCESSED_PATH = f'{DATA_DIR}/processed'
MODELS_DIR = f'{DATA_DIR}/models'

# Create necessary directories if they don't exist
os.makedirs(f'{DATA_DIR}/interim', exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Task 1: Ingest and Validate Data
def task_ingest_validate():
    print("--- STEP 1: Ingest & Validate ---")
    df = load_data(RAW_PATH)
    
    # Standardize column names (remove hidden spaces)
    df.columns = df.columns.str.strip()
    
    # Validate data schema and quality
    df = validate_input_data(df)
    
    # Save the validated data for the next step
    df.to_csv(STAGE_1_VALIDATED, index=False)
    print(f"Validated data saved to {STAGE_1_VALIDATED}")

# Task 2: Clean Data (Preprocessing)
def task_clean():
    print("--- STEP 2: Cleaning ---")
    df = pd.read_csv(STAGE_1_VALIDATED)
    
    # Apply cleaning logic (Imputation, etc.)
    df = clean_data(df)
    
    # Save cleaned data
    df.to_csv(STAGE_2_CLEANED, index=False)
    print(f"Cleaned data saved to {STAGE_2_CLEANED}")

# Task 3: Feature Engineering
def task_feature_eng():
    print("--- STEP 3: Feature Engineering ---")
    df = pd.read_csv(STAGE_2_CLEANED)
    
    # Apply Feature Crossing
    df = apply_feature_cross(df)
    
    # Save feature-engineered data
    df.to_csv(STAGE_3_FEATURES, index=False)
    print(f"Feature engineered data saved to {STAGE_3_FEATURES}")

# Task 4: Split, Balance, Hash and Save
def task_split_balance_save():
    print("--- STEP 4: Split, Balance, Hash & Save ---")
    df = pd.read_csv(STAGE_3_FEATURES)
    
    # Split Data into Train/Test sets
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Merge back to DataFrames for easier processing
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Balance Data (Applied ONLY to the Training Set to prevent data leakage)
    train_df = balance_data(train_df)
    
    # Apply Hashing Trick (For High Cardinality columns like Student_ID)
    # The same logic is applied to both Train and Test sets independently
    if 'Student_ID' in train_df.columns:
        train_df = apply_hashing(train_df, 'Student_ID', n_features=100)
    if 'Student_ID' in test_df.columns:
        test_df = apply_hashing(test_df, 'Student_ID', n_features=100)
    
    # Save Final Processed Artifacts
    # The training script (train_model.py) will read these files
    train_df.to_csv(f'{PROCESSED_PATH}/train_processed.csv', index=False)
    test_df.to_csv(f'{PROCESSED_PATH}/test_processed.csv', index=False)
    print("Processed files saved successfully. Ready for training.")

# Task 5: Model Training
def task_training():
    print("--- STEP 5: Training Model ---")
    # This calls the main function from src/train_model.py
    # It reads 'train_processed.csv' and saves 'model.pkl'
    train_model_main()
    print("Model training completed and saved to models/model.pkl")

# DAG Configuration
default_args = {
    'owner': 'Ege Karaurgan - MLOps Engineer',
    'depends_on_past': False,
    'retries': 0, # Fail fast for debugging
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'mlops_term_project_pipeline',
    default_args=default_args,
    description='End-to-End MLOps Pipeline (ETL + Training)',
    schedule_interval='@once', # Runs once when triggered manually
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['mlops', 'etl', 'training', 'docker']
) as dag:

    # Define Tasks (Airflow Operators)
    t1 = PythonOperator(
        task_id='1_ingest_and_validate',
        python_callable=task_ingest_validate
    )

    t2 = PythonOperator(
        task_id='2_clean_data',
        python_callable=task_clean
    )

    t3 = PythonOperator(
        task_id='3_feature_engineering',
        python_callable=task_feature_eng
    )

    t4 = PythonOperator(
        task_id='4_split_balance_save',
        python_callable=task_split_balance_save
    )

    t5 = PythonOperator(
        task_id='5_train_model',
        python_callable=task_training
    )

    # Define Dependencies (Linear Chain)
    # This creates the visual flow: t1 -> t2 -> t3 -> t4 -> t5
    t1 >> t2 >> t3 >> t4 >> t5

# Manual Execution Block (For testing via terminal without Airflow UI)
if __name__ == "__main__":
    print("🚀 Manual Execution Started for Testing...")
    try:
        task_ingest_validate()
        task_clean()
        task_feature_eng()
        task_split_balance_save()
        task_training()
        print("✅ All steps completed successfully!")
    except Exception as e:
        print(f"❌ Pipeline Failed: {e}")