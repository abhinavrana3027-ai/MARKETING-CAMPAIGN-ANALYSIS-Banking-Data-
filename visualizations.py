import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import DataLoader
from ml_model import ConversionPredictor

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

class ConversionVisualizer:
    """Create comprehensive visualizations for marketing analysis."""
    
    def __init__(self, df):
        self.df = df
    
    def plot_conversion_distribution(self):
        """Visualize overall conversion rates."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Conversion pie chart
        conversion_counts = (self.df['y'] == 'yes').value_counts()
        colors = ['#2ecc71', '#e74c3c']
        axes[0].pie(conversion_counts, labels=['No', 'Yes'], autopct='%1.1f%%',
                   colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
        axes[0].set_title('Overall Conversion Distribution', fontsize=14, weight='bold')
        
        # Count bar chart
        conversion_counts.plot(kind='bar', ax=axes[1], color=colors, edgecolor='black')
        axes[1].set_title('Conversion Counts', fontsize=14, weight='bold')
        axes[1].set_xlabel('Conversion Status')
        axes[1].set_ylabel('Count')
        axes[1].set_xticklabels(['Not Converted', 'Converted'], rotation=0)
        
        plt.tight_layout()
        plt.savefig('conversion_distribution.png', dpi=300, bbox_inches='tight')
        return fig
    
    def plot_demographic_analysis(self):
        """Visualize demographic patterns."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Age distribution by conversion
        self.df.boxplot(column='age', by='y', ax=axes[0, 0])
        axes[0, 0].set_title('Age Distribution by Conversion')
        axes[0, 0].set_xlabel('Conversion')
        axes[0, 0].set_ylabel('Age')
        
        # Job type
        job_conversion = self.df.groupby('job')['y'].apply(lambda x: (x == 'yes').sum() / len(x) * 100)
        job_conversion.sort_values(ascending=False).head(10).plot(kind='barh', ax=axes[0, 1], color='steelblue')
        axes[0, 1].set_title('Conversion Rate by Job Type')
        axes[0, 1].set_xlabel('Conversion Rate (%)')
        
        # Marital status
        marital_conversion = self.df.groupby('marital')['y'].apply(lambda x: (x == 'yes').sum() / len(x) * 100)
        marital_conversion.plot(kind='bar', ax=axes[1, 0], color='coral', edgecolor='black')
        axes[1, 0].set_title('Conversion Rate by Marital Status')
        axes[1, 0].set_ylabel('Conversion Rate (%)')
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)
        
        # Education
        edu_conversion = self.df.groupby('education')['y'].apply(lambda x: (x == 'yes').sum() / len(x) * 100)
        edu_conversion.plot(kind='bar', ax=axes[1, 1], color='lightgreen', edgecolor='black')
        axes[1, 1].set_title('Conversion Rate by Education')
        axes[1, 1].set_ylabel('Conversion Rate (%)')
        axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)
        
        plt.suptitle('')  # Remove automatic title
        plt.tight_layout()
        plt.savefig('demographic_analysis.png', dpi=300, bbox_inches='tight')
        return fig
    
    def plot_contact_analysis(self):
        """Visualize contact method and duration impact."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Contact type
        contact_conversion = self.df.groupby('contact')['y'].apply(lambda x: (x == 'yes').sum() / len(x) * 100)
        contact_conversion.plot(kind='bar', ax=axes[0, 0], color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Conversion Rate by Contact Type')
        axes[0, 0].set_ylabel('Conversion Rate (%)')
        axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=0)
        
        # Call duration (binned)
        duration_bins = [0, 180, 360, 600, 5000]
        duration_labels = ['0-3 min', '3-6 min', '6-10 min', '10+ min']
        self.df['duration_bin'] = pd.cut(self.df['duration'], bins=duration_bins, labels=duration_labels)
        duration_conversion = self.df.groupby('duration_bin')['y'].apply(lambda x: (x == 'yes').sum() / len(x) * 100)
        duration_conversion.plot(kind='bar', ax=axes[0, 1], color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Conversion Rate by Call Duration')
        axes[0, 1].set_ylabel('Conversion Rate (%)')
        axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)
        
        # Previous campaign
        prev_conversion = self.df.groupby('pdays')['y'].apply(lambda x: (x == 'yes').sum() / len(x) * 100)
        axes[1, 0].scatter(prev_conversion.index, prev_conversion.values, alpha=0.6, s=50)
        axes[1, 0].set_title('Conversion Rate vs Days Since Previous Contact')
        axes[1, 0].set_xlabel('Days')
        axes[1, 0].set_ylabel('Conversion Rate (%)')
        
        # Campaign attempts
        campaign_conversion = self.df.groupby('campaign')['y'].apply(lambda x: (x == 'yes').sum() / len(x) * 100)
        campaign_conversion.head(10).plot(kind='bar', ax=axes[1, 1], color='mediumpurple', edgecolor='black')
        axes[1, 1].set_title('Conversion Rate by Campaign Attempts')
        axes[1, 1].set_ylabel('Conversion Rate (%)')
        axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.savefig('contact_analysis.png', dpi=300, bbox_inches='tight')
        return fig
    
    def plot_temporal_analysis(self):
        """Visualize temporal patterns."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Month analysis
        month_conversion = self.df.groupby('month')['y'].apply(lambda x: (x == 'yes').sum() / len(x) * 100)
        month_conversion.plot(kind='bar', ax=axes[0, 0], color='gold', edgecolor='black')
        axes[0, 0].set_title('Conversion Rate by Month')
        axes[0, 0].set_ylabel('Conversion Rate (%)')
        axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)
        
        # Day of week
        day_conversion = self.df.groupby('day_of_week')['y'].apply(lambda x: (x == 'yes').sum() / len(x) * 100)
        day_conversion.plot(kind='bar', ax=axes[0, 1], color='lightseagreen', edgecolor='black')
        axes[0, 1].set_title('Conversion Rate by Day of Week')
        axes[0, 1].set_ylabel('Conversion Rate (%)')
        axes[0, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], rotation=0)
        
        # Quarterly analysis
        if 'quarter' in self.df.columns:
            quarter_conversion = self.df.groupby('quarter')['y'].apply(lambda x: (x == 'yes').sum() / len(x) * 100)
            quarter_conversion.plot(kind='bar', ax=axes[1, 0], color='tomato', edgecolor='black')
            axes[1, 0].set_title('Conversion Rate by Quarter')
            axes[1, 0].set_ylabel('Conversion Rate (%)')
        
        plt.tight_layout()
        plt.savefig('temporal_analysis.png', dpi=300, bbox_inches='tight')
        return fig


if __name__ == '__main__':
    loader = DataLoader('Bank.csv')
    df = loader.load_data()
    
    if df is not None:
        visualizer = ConversionVisualizer(df)
        
        print("Creating visualizations...")
        visualizer.plot_conversion_distribution()
        visualizer.plot_demographic_analysis()
        visualizer.plot_contact_analysis()
        visualizer.plot_temporal_analysis()
        
        print("\u2713 All visualizations saved successfully!")
