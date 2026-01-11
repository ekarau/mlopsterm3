import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

def clean_data(df):
    """
    Basic data cleaning and date formatting.
    """
    df = df.copy()
    
    # Target variable transformation (Target Encoding)
    if 'Completed' in df.columns:
        df['target'] = df['Completed'].apply(lambda x: 1 if x == 'Completed' else 0)
        df = df.drop(columns=['Completed'])
    
    # Date conversion
    if 'Enrollment_Date' in df.columns:
        df['Enrollment_Date'] = pd.to_datetime(df['Enrollment_Date'], dayfirst=True)
        # Deriving new feature from date (Month information)
        df['Enrollment_Month'] = df['Enrollment_Date'].dt.month
        df = df.drop(columns=['Enrollment_Date'])
        
    return df

def split_data(df):
    """
    Splits data into Train and Test sets.
    """
    X = df.drop(columns=['target'])
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def balance_data(train_df):
    """
    MANDATORY REQUIREMENT (III.2): Rebalancing Design Pattern.
    Addresses imbalance in training data using Upsampling.
    """
    print("Performing Rebalancing (Upsampling)...")
    majority = train_df[train_df.target == 0]
    minority = train_df[train_df.target == 1]
    
    # Minority class is upsampled to match the majority class
    if len(minority) < len(majority):
        minority_upsampled = resample(minority, 
                                      replace=True, 
                                      n_samples=len(majority), 
                                      random_state=42)
        return pd.concat([majority, minority_upsampled])
    
    return train_df