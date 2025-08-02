"""
Exploratory Data Analysis Module for Mental Health Risk Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MentalHealthEDA:
    """
    Comprehensive EDA for Mental Health Risk Prediction dataset
    """
    
    def __init__(self, output_dir="outputs/eda"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self, file_path):
        """Load the cleaned dataset"""
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def create_target_distribution_plot(self, df):
        """Create target variable distribution plot"""
        logger.info("Creating target distribution plot")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        target_counts = df['mental_health_risk'].value_counts()
        colors = ['#ff9999', '#66b3ff']
        labels = ['Low Risk', 'High Risk']
        
        ax1.pie(target_counts.values, labels=labels, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax1.set_title('Mental Health Risk Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart
        sns.countplot(data=df, x='mental_health_risk', ax=ax2, palette=colors)
        ax2.set_title('Mental Health Risk Count', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Risk Level (0=Low, 1=High)')
        ax2.set_ylabel('Count')
        
        # Add count labels on bars
        for i, v in enumerate(target_counts.values):
            ax2.text(i, v + 10, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/target_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Target distribution plot saved to {self.output_dir}/target_distribution.png")
    
    def create_age_analysis(self, df):
        """Create age-related analysis plots"""
        logger.info("Creating age analysis plots")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Age distribution by risk
        sns.histplot(data=df, x='Age', hue='mental_health_risk', bins=20, 
                    ax=ax1, palette=['#ff9999', '#66b3ff'], alpha=0.7)
        ax1.set_title('Age Distribution by Mental Health Risk', fontweight='bold')
        ax1.set_xlabel('Age')
        ax1.set_ylabel('Count')
        
        # Age boxplot by risk
        sns.boxplot(data=df, x='mental_health_risk', y='Age', ax=ax2, 
                   palette=['#ff9999', '#66b3ff'])
        ax2.set_title('Age Distribution by Risk Level', fontweight='bold')
        ax2.set_xlabel('Risk Level (0=Low, 1=High)')
        ax2.set_ylabel('Age')
        
        # Age groups analysis
        df['age_group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 100], 
                                labels=['18-25', '26-35', '36-45', '45+'])
        age_risk = df.groupby(['age_group', 'mental_health_risk']).size().unstack(fill_value=0)
        age_risk.plot(kind='bar', ax=ax3, color=['#ff9999', '#66b3ff'])
        ax3.set_title('Mental Health Risk by Age Group', fontweight='bold')
        ax3.set_xlabel('Age Group')
        ax3.set_ylabel('Count')
        ax3.legend(['Low Risk', 'High Risk'])
        ax3.tick_params(axis='x', rotation=45)
        
        # Risk percentage by age group
        age_risk_pct = age_risk.div(age_risk.sum(axis=1), axis=0) * 100
        age_risk_pct.plot(kind='bar', ax=ax4, color=['#ff9999', '#66b3ff'])
        ax4.set_title('Risk Percentage by Age Group', fontweight='bold')
        ax4.set_xlabel('Age Group')
        ax4.set_ylabel('Percentage (%)')
        ax4.legend(['Low Risk', 'High Risk'])
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/age_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Age analysis plots saved to {self.output_dir}/age_analysis.png")
    
    def create_gender_analysis(self, df):
        """Create gender-related analysis plots"""
        logger.info("Creating gender analysis plots")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Gender distribution
        gender_counts = df['Gender'].value_counts()
        ax1.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
                startangle=90)
        ax1.set_title('Gender Distribution', fontweight='bold')
        
        # Gender vs Risk
        gender_risk = df.groupby(['Gender', 'mental_health_risk']).size().unstack(fill_value=0)
        gender_risk.plot(kind='bar', ax=ax2, color=['#ff9999', '#66b3ff'])
        ax2.set_title('Mental Health Risk by Gender', fontweight='bold')
        ax2.set_xlabel('Gender')
        ax2.set_ylabel('Count')
        ax2.legend(['Low Risk', 'High Risk'])
        ax2.tick_params(axis='x', rotation=45)
        
        # Risk percentage by gender
        gender_risk_pct = gender_risk.div(gender_risk.sum(axis=1), axis=0) * 100
        gender_risk_pct.plot(kind='bar', ax=ax3, color=['#ff9999', '#66b3ff'])
        ax3.set_title('Risk Percentage by Gender', fontweight='bold')
        ax3.set_xlabel('Gender')
        ax3.set_ylabel('Percentage (%)')
        ax3.legend(['Low Risk', 'High Risk'])
        ax3.tick_params(axis='x', rotation=45)
        
        # Gender and age interaction
        sns.boxplot(data=df, x='Gender', y='Age', hue='mental_health_risk', ax=ax4)
        ax4.set_title('Age Distribution by Gender and Risk', fontweight='bold')
        ax4.set_xlabel('Gender')
        ax4.set_ylabel('Age')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/gender_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gender analysis plots saved to {self.output_dir}/gender_analysis.png")
    
    def create_work_related_analysis(self, df):
        """Create work-related analysis plots"""
        logger.info("Creating work-related analysis plots")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Remote work analysis
        remote_risk = df.groupby(['is_remote_worker', 'mental_health_risk']).size().unstack(fill_value=0)
        remote_risk.plot(kind='bar', ax=ax1, color=['#ff9999', '#66b3ff'])
        ax1.set_title('Mental Health Risk by Remote Work Status', fontweight='bold')
        ax1.set_xlabel('Remote Worker (0=No, 1=Yes)')
        ax1.set_ylabel('Count')
        ax1.legend(['Low Risk', 'High Risk'])
        
        # Work interference analysis
        if 'work_interfere' in df.columns:
            work_interfere_risk = df.groupby(['work_interfere', 'mental_health_risk']).size().unstack(fill_value=0)
            work_interfere_risk.plot(kind='bar', ax=ax2, color=['#ff9999', '#66b3ff'])
            ax2.set_title('Mental Health Risk by Work Interference', fontweight='bold')
            ax2.set_xlabel('Work Interference Level')
            ax2.set_ylabel('Count')
            ax2.legend(['Low Risk', 'High Risk'])
            ax2.tick_params(axis='x', rotation=45)
        
        # Company size analysis
        if 'company_size_category' in df.columns:
            company_risk = df.groupby(['company_size_category', 'mental_health_risk']).size().unstack(fill_value=0)
            company_risk.plot(kind='bar', ax=ax3, color=['#ff9999', '#66b3ff'])
            ax3.set_title('Mental Health Risk by Company Size', fontweight='bold')
            ax3.set_xlabel('Company Size Category')
            ax3.set_ylabel('Count')
            ax3.legend(['Low Risk', 'High Risk'])
            ax3.tick_params(axis='x', rotation=45)
        
        # Work stress level analysis
        if 'work_stress_level' in df.columns:
            stress_risk = df.groupby(['work_stress_level', 'mental_health_risk']).size().unstack(fill_value=0)
            stress_risk.plot(kind='bar', ax=ax4, color=['#ff9999', '#66b3ff'])
            ax4.set_title('Mental Health Risk by Work Stress Level', fontweight='bold')
            ax4.set_xlabel('Work Stress Level')
            ax4.set_ylabel('Count')
            ax4.legend(['Low Risk', 'High Risk'])
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/work_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Work analysis plots saved to {self.output_dir}/work_analysis.png")
    
    def create_correlation_heatmap(self, df):
        """Create correlation heatmap"""
        logger.info("Creating correlation heatmap")
        
        # Select numerical columns for correlation
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numerical_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Correlation heatmap saved to {self.output_dir}/correlation_heatmap.png")
        
        return correlation_matrix
    
    def create_feature_importance_plot(self, df):
        """Create feature importance plot based on correlation with target"""
        logger.info("Creating feature importance plot")
        
        # Calculate correlation with target
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        target_correlations = df[numerical_cols].corr()['mental_health_risk'].abs().sort_values(ascending=False)
        
        # Remove target from correlations
        target_correlations = target_correlations.drop('mental_health_risk')
        
        plt.figure(figsize=(12, 8))
        colors = ['#ff6b6b' if x > 0.1 else '#4ecdc4' for x in target_correlations.values]
        bars = plt.barh(range(len(target_correlations)), target_correlations.values, color=colors)
        plt.yticks(range(len(target_correlations)), target_correlations.index)
        plt.xlabel('Absolute Correlation with Target')
        plt.title('Feature Importance (Correlation with Mental Health Risk)', fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, target_correlations.values)):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {self.output_dir}/feature_importance.png")
        
        return target_correlations
    
    def create_interactive_dashboard(self, df):
        """Create interactive Plotly dashboard"""
        logger.info("Creating interactive dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Age vs Risk by Gender', 'Work Stress vs Risk', 
                          'Remote Work Impact', 'Company Size vs Risk'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Age vs Risk by Gender
        for gender in df['Gender'].unique():
            gender_data = df[df['Gender'] == gender]
            fig.add_trace(
                go.Scatter(x=gender_data['Age'], y=gender_data['mental_health_risk'],
                          mode='markers', name=f'Gender {gender}', opacity=0.7),
                row=1, col=1
            )
        
        # Work Stress vs Risk
        if 'work_stress_level' in df.columns:
            stress_risk = df.groupby('work_stress_level')['mental_health_risk'].mean()
            fig.add_trace(
                go.Bar(x=stress_risk.index, y=stress_risk.values, name='Risk Rate'),
                row=1, col=2
            )
        
        # Remote Work Impact
        remote_risk = df.groupby('is_remote_worker')['mental_health_risk'].mean()
        fig.add_trace(
            go.Bar(x=['Not Remote', 'Remote'], y=remote_risk.values, name='Risk Rate'),
            row=2, col=1
        )
        
        # Company Size vs Risk
        if 'company_size_category' in df.columns:
            company_risk = df.groupby('company_size_category')['mental_health_risk'].mean()
            fig.add_trace(
                go.Bar(x=company_risk.index, y=company_risk.values, name='Risk Rate'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Mental Health Risk Analysis Dashboard")
        fig.write_html(f"{self.output_dir}/interactive_dashboard.html")
        
        logger.info(f"Interactive dashboard saved to {self.output_dir}/interactive_dashboard.html")
    
    def generate_eda_report(self, df):
        """Generate comprehensive EDA report"""
        logger.info("Generating comprehensive EDA report")
        
        # Create all visualizations
        self.create_target_distribution_plot(df)
        self.create_age_analysis(df)
        self.create_gender_analysis(df)
        self.create_work_related_analysis(df)
        correlation_matrix = self.create_correlation_heatmap(df)
        feature_importance = self.create_feature_importance_plot(df)
        self.create_interactive_dashboard(df)
        
        # Generate summary statistics
        summary_stats = {
            'total_samples': len(df),
            'high_risk_percentage': (df['mental_health_risk'] == 1).mean() * 100,
            'average_age': df['Age'].mean(),
            'remote_worker_percentage': df['is_remote_worker'].mean() * 100,
            'top_features': feature_importance.head(5).index.tolist()
        }
        
        # Save summary to file
        with open(f"{self.output_dir}/eda_summary.txt", 'w') as f:
            f.write("MENTAL HEALTH RISK PREDICTION - EDA SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total samples: {summary_stats['total_samples']}\n")
            f.write(f"High risk percentage: {summary_stats['high_risk_percentage']:.2f}%\n")
            f.write(f"Average age: {summary_stats['average_age']:.1f} years\n")
            f.write(f"Remote worker percentage: {summary_stats['remote_worker_percentage']:.2f}%\n")
            f.write(f"Top 5 most important features: {', '.join(summary_stats['top_features'])}\n")
        
        logger.info(f"EDA summary saved to {self.output_dir}/eda_summary.txt")
        logger.info("Comprehensive EDA completed successfully!")
        
        return summary_stats

if __name__ == "__main__":
    # Example usage
    eda = MentalHealthEDA()
    df = eda.load_data("data/processed/cleaned_data.csv")
    summary = eda.generate_eda_report(df)
    print("EDA completed successfully!") 