import pandas as pd
from sklearn.feature_extraction import FeatureHasher

def apply_feature_cross(df):
    """
    MANDATORY REQUIREMENT (III.1): Feature Interactions / Cross.
    Creates a new feature by combining Category and Course Level.
    """
    print("Applying Feature Cross...")
    # Example: 'Programming' + 'Beginner' -> 'Programming_Beginner'
    df['Category_Level_Cross'] = df['Category'] + '_' + df['Course_Level']
    return df

def apply_hashing(df, col_name, n_features=100):
    """
    MANDATORY REQUIREMENT (III.1): High Cardinality Handling (Hashing Trick).
    Compresses columns with high unique values (e.g., Student_ID).
    """
    print(f"Applying Hashing Trick: {col_name} -> {n_features} buckets")
    
    # FeatureHasher expects a list of lists of strings: [['id1'], ['id2'], ...]
    col_data = [[str(x)] for x in df[col_name]]
    
    hasher = FeatureHasher(n_features=n_features, input_type='string')
    hashed_features = hasher.transform(col_data).toarray()
    
    # Hashed column names
    hashed_cols = [f'hashed_{col_name}_{i}' for i in range(n_features)]
    hashed_df = pd.DataFrame(hashed_features, columns=hashed_cols, index=df.index)
    
    # Dropping the original high-cardinality column and adding the hashed version
    return pd.concat([df.drop(columns=[col_name]), hashed_df], axis=1)