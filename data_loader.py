import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure plotting
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

class DataLoader:
    """Load and preprocess marketing campaign data."""
    
    def __init__(self, data_path='Bank.csv'):
        self.data_path = data_path
        self.df = None
    
    def load_data(self):
        """Load CSV data file."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Data loaded successfully!")
            print(f"  Shape: {self.df.shape}")
            print(f"  Columns: {list(self.df.columns)}")
            return self.df
        except FileNotFoundError:
            print(f"✗ Error: File '{self.data_path}' not found")
            return None
    
    def get_basic_info(self):
        """Display basic information about the dataset."""
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        print("\n=== Dataset Overview ===")
        print(self.df.info())
        print("\n=== Statistical Summary ===")
        print(self.df.describe())
        print("\n=== Missing Values ===")
        print(self.df.isnull().sum())
    
    def get_head(self, n=5):
        """Display first n rows."""
        if self.df is None:
            print("No data loaded.")
            return
        return self.df.head(n)
    
    def get_shape(self):
        """Get dataset shape."""
        if self.df is None:
            return None
        return self.df.shape


if __name__ == '__main__':
    # Load data
    loader = DataLoader('Bank.csv')
    df = loader.load_data()
    
    # Display info
    if df is not None:
        loader.get_basic_info()
        print("\n=== First 5 Rows ===")
        print(loader.get_head())
