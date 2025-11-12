import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import DataLoader

# Configure visualization
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

class MarketingAnalyzer:
    """Analyze marketing campaign data."""
    
    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.categorical_cols = df.select_dtypes(include=['object']).columns
    
    def conversion_analysis(self):
        """Analyze conversion rates."""
        if 'y' in self.df.columns:
            conversion = (self.df['y'] == 'yes').sum()
            total = len(self.df)
            rate = (conversion / total) * 100
            
            print("\n=== Conversion Analysis ===")
            print(f"Total Contacts: {total}")
            print(f"Conversions: {conversion}")
            print(f"Conversion Rate: {rate:.2f}%")
            
            return {'total': total, 'conversions': conversion, 'rate': rate}
    
    def demographic_analysis(self):
        """Analyze demographic patterns."""
        print("\n=== Demographic Analysis ===")
        
        if 'age' in self.df.columns:
            print(f"\nAge Statistics:")
            print(f"  Mean: {self.df['age'].mean():.1f}")
            print(f"  Median: {self.df['age'].median():.1f}")
            print(f"  Min: {self.df['age'].min()}")
            print(f"  Max: {self.df['age'].max()}")
        
        if 'job' in self.df.columns:
            print(f"\nTop 5 Job Categories:")
            print(self.df['job'].value_counts().head())
    
    def contact_analysis(self):
        """Analyze contact patterns."""
        print("\n=== Contact Analysis ===")
        
        if 'contact' in self.df.columns:
            print(f"Contact Methods:")
            print(self.df['contact'].value_counts())
        
        if 'duration' in self.df.columns:
            print(f"\nCall Duration Statistics (seconds):")
            print(f"  Mean: {self.df['duration'].mean():.1f}")
            print(f"  Median: {self.df['duration'].median():.1f}")
            print(f"  Max: {self.df['duration'].max()}")
    
    def visualize_distributions(self, output_dir='results/'):
        """Create distribution visualizations."""
        # Numeric distributions
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        numeric_cols_sample = list(self.numeric_cols[:4])
        for idx, col in enumerate(numeric_cols_sample):
            row = idx // 2
            col_idx = idx % 2
            axes[row, col_idx].hist(self.df[col], bins=50, edgecolor='black', alpha=0.7)
            axes[row, col_idx].set_title(f'Distribution of {col}')
            axes[row, col_idx].set_xlabel(col)
            axes[row, col_idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        return fig
    
    def generate_report(self):
        """Generate comprehensive analysis report."""
        print("\n" + "="*50)
        print("MARKETING CAMPAIGN ANALYSIS REPORT")
        print("="*50)
        
        self.conversion_analysis()
        self.demographic_analysis()
        self.contact_analysis()
        
        print("\n" + "="*50)


if __name__ == '__main__':
    # Load data
    loader = DataLoader('Bank.csv')
    df = loader.load_data()
    
    if df is not None:
        # Run analysis
        analyzer = MarketingAnalyzer(df)
        analyzer.generate_report()
        
        # Generate visualizations
        fig = analyzer.visualize_distributions()
        plt.savefig('results/distributions.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualizations saved to results/")
