import pandas as pd
import os

def load_data(file_path):
    """
    Reads the CSV file from the specified path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ERROR: File not found -> {file_path}")
    
    print(f"Loading data: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Data loaded. Shape: {df.shape}")
    return df