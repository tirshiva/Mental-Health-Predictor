"""
Model Explainability Module using SHAP for Mental Health Risk Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MentalHealthModelExplainer:
    """
    SHAP-based model explainability for Mental Health Risk Prediction
    """
    
    def __init__(self, output_dir="outputs/shap"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        self.model = None
        self.scaler = None
        
    def load_model_and_data(self, model_path, scaler_path, feature_names_path, data_path):
        """Load trained model, scaler, and data"""
        logger.info("Loading model and data for explainability analysis")
        
        # Load model
        self.model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Load scaler
        self.scaler = joblib.load(scaler_path)
        logger.info(f"Loaded scaler from {scaler_path}")
        
        # Load feature names
        with open(feature_names_path, 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        logger.info(f"Loaded {len(self.feature_names)} feature names")
        
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data with {len(df)} samples")
        
        return df
    
    def prepare_data_for_shap(self, df):
        """Prepare data for SHAP analysis"""
        logger.info("Preparing data for SHAP analysis")
        
        # Separate features and target
        X = df.drop('mental_health_risk', axis=1)
        y = df['mental_health_risk']
        
        # Ensure columns match feature names
        if self.feature_names:
            X = X[self.feature_names]
        
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        logger.info(f"Prepared {X_scaled.shape[0]} samples with {X_scaled.shape[1]} features")
        
        return X_scaled, y
    
    def create_shap_explainer(self, X_sample):
        """Create SHAP explainer"""
        logger.info("Creating SHAP explainer")
        
        # Create explainer based on model type
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models (Random Forest, XGBoost)
            self.explainer = shap.TreeExplainer(self.model)
        else:
            # Linear models (Logistic Regression)
            self.explainer = shap.LinearExplainer(self.model, X_sample)
        
        logger.info("SHAP explainer created successfully")
    
    def calculate_shap_values(self, X):
        """Calculate SHAP values"""
        logger.info("Calculating SHAP values")
        
        # Use a sample for background if needed
        if len(X) > 100:
            background_sample = X[:100]
        else:
            background_sample = X
        
        # Calculate SHAP values
        self.shap_values = self.explainer.shap_values(X)
        
        # Handle different output formats
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # For binary classification
        
        logger.info(f"Calculated SHAP values with shape: {self.shap_values.shape}")
        
        return self.shap_values
    
    def create_summary_plot(self, X, max_display=20):
        """Create SHAP summary plot"""
        logger.info("Creating SHAP summary plot")
        
        plt.figure(figsize=(12, 8))
        
        # Create summary plot
        shap.summary_plot(
            self.shap_values, 
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        plt.title('SHAP Summary Plot - Feature Importance', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/shap_summary_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP summary plot saved to {self.output_dir}/shap_summary_plot.png")
    
    def create_bar_plot(self, X):
        """Create SHAP bar plot"""
        logger.info("Creating SHAP bar plot")
        
        plt.figure(figsize=(12, 8))
        
        # Create bar plot
        shap.summary_plot(
            self.shap_values, 
            X,
            feature_names=self.feature_names,
            plot_type="bar",
            show=False
        )
        
        plt.title('SHAP Feature Importance (Bar Plot)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/shap_bar_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP bar plot saved to {self.output_dir}/shap_bar_plot.png")
    
    def create_waterfall_plots(self, X, y, num_samples=10):
        """Create waterfall plots for individual predictions"""
        logger.info("Creating waterfall plots for individual predictions")
        
        # Select samples for waterfall plots
        high_risk_indices = np.where(y == 1)[0][:num_samples//2]
        low_risk_indices = np.where(y == 0)[0][:num_samples//2]
        selected_indices = np.concatenate([high_risk_indices, low_risk_indices])
        
        for i, idx in enumerate(selected_indices):
            plt.figure(figsize=(12, 8))
            
            # Create waterfall plot
            shap.waterfall_plot(
                shap.Explanation(
                    values=self.shap_values[idx],
                    base_values=self.explainer.expected_value,
                    data=X[idx],
                    feature_names=self.feature_names
                ),
                show=False
            )
            
            risk_status = "High Risk" if y[idx] == 1 else "Low Risk"
            plt.title(f'SHAP Waterfall Plot - Sample {idx} ({risk_status})', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/waterfall_plot_sample_{idx}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Created {len(selected_indices)} waterfall plots")
    
    def create_force_plots(self, X, y, num_samples=5):
        """Create force plots for individual predictions"""
        logger.info("Creating force plots for individual predictions")
        
        # Select samples for force plots
        high_risk_indices = np.where(y == 1)[0][:num_samples//2]
        low_risk_indices = np.where(y == 0)[0][:num_samples//2]
        selected_indices = np.concatenate([high_risk_indices, low_risk_indices])
        
        for i, idx in enumerate(selected_indices):
            plt.figure(figsize=(12, 6))
            
            # Create force plot
            shap.force_plot(
                self.explainer.expected_value,
                self.shap_values[idx],
                X[idx],
                feature_names=self.feature_names,
                show=False
            )
            
            risk_status = "High Risk" if y[idx] == 1 else "Low Risk"
            plt.title(f'SHAP Force Plot - Sample {idx} ({risk_status})', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/force_plot_sample_{idx}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Created {len(selected_indices)} force plots")
    
    def create_dependence_plots(self, X, top_features=5):
        """Create dependence plots for top features"""
        logger.info("Creating SHAP dependence plots")
        
        # Get feature importance
        feature_importance = np.abs(self.shap_values).mean(0)
        top_feature_indices = np.argsort(feature_importance)[-top_features:]
        
        for i, feature_idx in enumerate(top_feature_indices):
            feature_name = self.feature_names[feature_idx]
            
            plt.figure(figsize=(10, 6))
            
            # Create dependence plot
            shap.dependence_plot(
                feature_idx,
                self.shap_values,
                X,
                feature_names=self.feature_names,
                show=False
            )
            
            plt.title(f'SHAP Dependence Plot - {feature_name}', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/dependence_plot_{feature_name}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Created {top_features} dependence plots")
    
    def create_interaction_plots(self, X, top_features=3):
        """Create interaction plots for top features"""
        logger.info("Creating SHAP interaction plots")
        
        # Get feature importance
        feature_importance = np.abs(self.shap_values).mean(0)
        top_feature_indices = np.argsort(feature_importance)[-top_features:]
        
        for i, feature_idx in enumerate(top_feature_indices):
            feature_name = self.feature_names[feature_idx]
            
            # Find the second most important feature for interaction
            other_features = [j for j in range(len(self.feature_names)) if j != feature_idx]
            other_importance = feature_importance[other_features]
            second_feature_idx = other_features[np.argmax(other_importance)]
            second_feature_name = self.feature_names[second_feature_idx]
            
            plt.figure(figsize=(10, 6))
            
            # Create interaction plot
            shap.dependence_plot(
                feature_idx,
                self.shap_values,
                X,
                feature_names=self.feature_names,
                interaction_index=second_feature_idx,
                show=False
            )
            
            plt.title(f'SHAP Interaction Plot - {feature_name} vs {second_feature_name}', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/interaction_plot_{feature_name}_{second_feature_name}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Created {top_features} interaction plots")
    
    def create_feature_importance_analysis(self, X):
        """Create comprehensive feature importance analysis"""
        logger.info("Creating feature importance analysis")
        
        # Calculate mean absolute SHAP values
        mean_shap_values = np.abs(self.shap_values).mean(0)
        
        # Create feature importance DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_shap_values
        }).sort_values('importance', ascending=False)
        
        # Save feature importance
        feature_importance_df.to_csv(f"{self.output_dir}/feature_importance_shap.csv", index=False)
        
        # Create feature importance plot
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(15)
        
        colors = ['#ff6b6b' if x > np.median(top_features['importance']) else '#4ecdc4' 
                 for x in top_features['importance']]
        
        bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Feature Importance Based on SHAP Values', fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{value:.4f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/feature_importance_shap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance analysis saved to {self.output_dir}")
        
        return feature_importance_df
    
    def generate_explainability_report(self, feature_importance_df):
        """Generate comprehensive explainability report"""
        logger.info("Generating explainability report")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{self.output_dir}/shap_explainability_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("MENTAL HEALTH RISK PREDICTION - SHAP EXPLAINABILITY REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("TOP 10 MOST IMPORTANT FEATURES:\n")
            f.write("-" * 35 + "\n")
            for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows()):
                f.write(f"{i+1:2d}. {row['feature']:<30} {row['importance']:.4f}\n")
            
            f.write(f"\nTotal features analyzed: {len(feature_importance_df)}\n")
            f.write(f"SHAP analysis completed successfully!\n")
        
        logger.info(f"Explainability report saved to {report_path}")
    
    def run_complete_analysis(self, model_path, scaler_path, feature_names_path, data_path):
        """Run complete SHAP explainability analysis"""
        logger.info("Starting complete SHAP explainability analysis")
        
        # Load model and data
        df = self.load_model_and_data(model_path, scaler_path, feature_names_path, data_path)
        
        # Prepare data
        X, y = self.prepare_data_for_shap(df)
        
        # Create explainer
        self.create_shap_explainer(X)
        
        # Calculate SHAP values
        self.calculate_shap_values(X)
        
        # Create all visualizations
        self.create_summary_plot(X)
        self.create_bar_plot(X)
        self.create_waterfall_plots(X, y)
        self.create_force_plots(X, y)
        self.create_dependence_plots(X)
        self.create_interaction_plots(X)
        
        # Create feature importance analysis
        feature_importance_df = self.create_feature_importance_analysis(X)
        
        # Generate report
        self.generate_explainability_report(feature_importance_df)
        
        logger.info("Complete SHAP explainability analysis finished successfully!")
        
        return feature_importance_df

if __name__ == "__main__":
    # Example usage
    explainer = MentalHealthModelExplainer()
    
    # You would need to provide the actual paths to your trained model files
    # feature_importance_df = explainer.run_complete_analysis(
    #     model_path="models/best_model_20231201_120000.joblib",
    #     scaler_path="models/scaler_20231201_120000.joblib",
    #     feature_names_path="models/feature_names_20231201_120000.txt",
    #     data_path="data/processed/cleaned_data.csv"
    # )
    
    print("SHAP explainability module ready!") 