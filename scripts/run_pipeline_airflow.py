import pandas as pd
import os
import sys

# Adding path so Python can find the 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Importing functions from our own modules
from src.ingest import load_data
from src.validate import validate_input_data  # Added for consistency with DAG
from src.preprocess import clean_data, split_data, balance_data
from src.features import apply_feature_cross, apply_hashing

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'Course_Completion_Prediction.csv')
    PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    
    print("🚀 Starting Pipeline (Local Mode)...")

    # 2. INGEST (DATA LOADING)
    try:
        df = load_data(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"❌ ERROR: File '{RAW_DATA_PATH}' not found.")
        print("Please ensure the CSV file is placed in the 'data/raw' folder.")
        return

    # 2.5 VALIDATION (Added for consistency)
    print("🔍 Validating input data...")
    try:
        df = validate_input_data(df)
    except ValueError as e:
        print(e)
        return

    # 3. PREPROCESS
    print("🧹 Cleaning data...")
    df = clean_data(df)

    # 4. FEATURE ENGINEERING - CROSS
    print("X  Applying Feature Cross...")
    df = apply_feature_cross(df)

    # 5. DATA SPLITTING
    print("✂️  Splitting data into Train/Test...")
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Merging as DataFrame (For ease of processing)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # 6. REBALANCING (Only applied to Training Data!)
    print("⚖️  Balancing training data (Upsampling)...")
    train_df = balance_data(train_df)

    # 7. HASHING (HIGH CARDINALITY)
    # Hashing Student_ID column to convert it to numerical format
    print("Processing Hashing (Student_ID)...")
    train_df = apply_hashing(train_df, 'Student_ID', n_features=100)
    test_df = apply_hashing(test_df, 'Student_ID', n_features=100)

    # 8. SAVING
    print("💾 Saving files...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    train_path = os.path.join(PROCESSED_DIR, 'train_processed.csv')
    test_path = os.path.join(PROCESSED_DIR, 'test_processed.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✅ PROCESS SUCCESSFUL!")
    print(f"   -> Created file: {train_path}")
    print(f"   -> Created file: {test_path}")

if __name__ == "__main__":
    main()