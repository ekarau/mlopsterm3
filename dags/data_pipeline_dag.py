from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.ingest import load_data
from src.validate import validate_input_data
from src.preprocess import clean_data, split_data, balance_data
from src.features import apply_feature_cross, apply_hashing
from src.train_model import main as train_model_main
import pandas as pd
import sys
import os

sys.path.append('/opt/airflow')

DATA_DIR = '/opt/airflow/data'
RAW_PATH = f'{DATA_DIR}/raw/Course_Completion_Prediction.csv'
STAGE_1_VALIDATED = f'{DATA_DIR}/interim/1_validated.csv'
STAGE_2_CLEANED = f'{DATA_DIR}/interim/2_cleaned.csv'
STAGE_3_FEATURES = f'{DATA_DIR}/interim/3_features.csv'
PROCESSED_PATH = f'{DATA_DIR}/processed'
MODELS_DIR = f'{DATA_DIR}/models'

os.makedirs(f'{DATA_DIR}/interim', exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def task_ingest_validate():
    print("--- STEP 1: Ingest & Validate ---")
    df = load_data(RAW_PATH)

    df.columns = df.columns.str.strip()

    df = validate_input_data(df)

    df.to_csv(STAGE_1_VALIDATED, index=False)
    print(f"Validated data saved to {STAGE_1_VALIDATED}")


def task_clean():
    print("--- STEP 2: Cleaning ---")
    df = pd.read_csv(STAGE_1_VALIDATED)

    df = clean_data(df)

    df.to_csv(STAGE_2_CLEANED, index=False)
    print(f"Cleaned data saved to {STAGE_2_CLEANED}")


def task_feature_eng():
    print("--- STEP 3: Feature Engineering ---")
    df = pd.read_csv(STAGE_2_CLEANED)

    df = apply_feature_cross(df)

    df.to_csv(STAGE_3_FEATURES, index=False)
    print(f"Feature engineered data saved to {STAGE_3_FEATURES}")


def task_split_balance_save():
    print("--- STEP 4: Split, Balance, Hash & Save ---")
    df = pd.read_csv(STAGE_3_FEATURES)

    X_train, X_test, y_train, y_test = split_data(df)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df = balance_data(train_df)

    if 'Student_ID' in train_df.columns:
        train_df = apply_hashing(train_df, 'Student_ID', n_features=100)
    if 'Student_ID' in test_df.columns:
        test_df = apply_hashing(test_df, 'Student_ID', n_features=100)

    train_df.to_csv(f'{PROCESSED_PATH}/train_processed.csv', index=False)
    test_df.to_csv(f'{PROCESSED_PATH}/test_processed.csv', index=False)
    print("Processed files saved successfully. Ready for training.")


def task_training():
    print("--- STEP 5: Training Model ---")
    train_model_main()
    print("Model training completed and saved to models/model.pkl")


default_args = {
    'owner': 'Ege Karaurgan - MLOps Engineer',
    'depends_on_past': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'mlops_term_project_pipeline',
    default_args=default_args,
    description='End-to-End MLOps Pipeline (ETL + Training)',
    schedule_interval='@once',
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['mlops', 'etl', 'training', 'docker']
) as dag:

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

    t1 >> t2 >> t3 >> t4 >> t5

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
