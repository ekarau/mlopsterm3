import unittest
import pandas as pd
import sys
import os

# Add src folder to path (so tests can see src)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import apply_feature_cross, apply_hashing

class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        """Dummy dataset to run before each test."""
        self.df = pd.DataFrame({
            'Student_ID': ['S1', 'S2', 'S3', 'S1'],
            'Category': ['Programming', 'Math', 'Science', 'Programming'],
            'Course_Level': ['Beginner', 'Intermediate', 'Beginner', 'Advanced']
        })

    def test_feature_cross(self):
        """Tests the correctness of the Feature Cross operation."""
        df_crossed = apply_feature_cross(self.df.copy())
        
        # 1. Was the new column created?
        self.assertIn('Category_Level_Cross', df_crossed.columns)
        
        # 2. Did the values merge correctly? (Programming + Beginner -> Programming_Beginner)
        expected_value = 'Programming_Beginner'
        self.assertEqual(df_crossed.iloc[0]['Category_Level_Cross'], expected_value)

    def test_hashing_trick(self):
        """Tests that the Hashing operation transforms columns correctly."""
        n_features = 5
        col_name = 'Student_ID'
        
        df_hashed = apply_hashing(self.df.copy(), col_name, n_features=n_features)
        
        # 1. Was the original column dropped?
        self.assertNotIn(col_name, df_hashed.columns)
        
        # 2. Were the hash columns created?
        hash_cols = [col for col in df_hashed.columns if 'hashed_' in col]
        self.assertEqual(len(hash_cols), n_features)
        
        # 3. Did the row count remain the same without data loss?
        self.assertEqual(len(df_hashed), len(self.df))

if __name__ == '__main__':
    unittest.main()