import pandas as pd

def validate_input_data(df):
    """
    MANDATORY REQUIREMENT (III.3): Monitoring & Statistical Checks.
    Checks data quality using logic similar to Great Expectations.
    Stops the pipeline in case of error.
    """
    print("🔍 Starting Data Validation...")
    
    # 1. Mandatory Column Check (Schema Check)
    required_columns = ['Student_ID', 'Category', 'Course_Level', 'Completed']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"❌ ERROR: Missing columns: {missing_cols}")
    
    # 2. Null Value Check (Completeness Check)
    # Critical columns must not contain null values
    if df[required_columns].isnull().any().any():
        null_counts = df[required_columns].isnull().sum()
        raise ValueError(f"❌ ERROR: Null values (NaN) found in critical columns:\n{null_counts}")
        
    # 3. Cardinality/Size Check (Statistical Check)
    # Example: Raise error if the dataset is empty
    if len(df) == 0:
        raise ValueError("❌ ERROR: Dataset is empty!")
        
    print("✅ Validation Successful: Data schema and quality are valid.")
    return df