"""
Model Training Module for Mental Health Risk Prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MentalHealthModelTrainer:
    """
    Comprehensive model trainer for Mental Health Risk Prediction
    """
    
    def __init__(self, models_dir="models", output_dir="outputs"):
        self.models_dir = models_dir
        self.output_dir = output_dir
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, file_path):
        """Load the cleaned dataset"""
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def prepare_data(self, df):
        """Prepare data for training"""
        logger.info("Preparing data for training")
        
        # Separate features and target
        X = df.drop('mental_health_risk', axis=1)
        y = df['mental_health_risk']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        logger.info(f"Number of features: {X_train.shape[1]}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    def define_models(self):
        """Define the models to train"""
        logger.info("Defining models for training")
        
        self.models = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }
        }
        
        logger.info(f"Defined {len(self.models)} models for training")
    
    def train_models(self, X_train, y_train):
        """Train all models with hyperparameter tuning"""
        logger.info("Starting model training with hyperparameter tuning")
        
        results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model_info in self.models.items():
            logger.info(f"Training {name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                estimator=model_info['model'],
                param_grid=model_info['params'],
                cv=cv,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Store results
            results[name] = {
                'best_model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            logger.info(f"{name} - Best F1 Score: {grid_search.best_score_:.4f}")
            logger.info(f"{name} - Best Parameters: {grid_search.best_params_}")
        
        return results
    
    def evaluate_models(self, results, X_test, y_test):
        """Evaluate all trained models"""
        logger.info("Evaluating models on test set")
        
        evaluation_results = {}
        
        for name, model_info in results.items():
            model = model_info['best_model']
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            evaluation_results[name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            logger.info(f"{name} Test Results:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return evaluation_results
    
    def select_best_model(self, evaluation_results):
        """Select the best model based on F1-score"""
        logger.info("Selecting best model")
        
        best_score = 0
        best_model_name = None
        
        for name, results in evaluation_results.items():
            f1_score = results['metrics']['f1_score']
            if f1_score > best_score:
                best_score = f1_score
                best_model_name = name
        
        self.best_model = evaluation_results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        logger.info(f"Best model: {best_model_name} with F1-score: {best_score:.4f}")
        
        return best_model_name, evaluation_results[best_model_name]
    
    def create_model_comparison_plot(self, evaluation_results):
        """Create comparison plot of all models"""
        logger.info("Creating model comparison plot")
        
        # Extract metrics for plotting
        models = list(evaluation_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Create comparison data
        comparison_data = []
        for model in models:
            model_metrics = evaluation_results[model]['metrics']
            comparison_data.append([model_metrics[metric] for metric in metrics])
        
        comparison_df = pd.DataFrame(comparison_data, index=models, columns=metrics)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        comparison_df.plot(kind='bar', ax=ax, colormap='viridis')
        ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f"{self.output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison plot saved to {self.output_dir}/model_comparison.png")
        
        return comparison_df
    
    def create_confusion_matrices(self, evaluation_results, y_test):
        """Create confusion matrices for all models"""
        logger.info("Creating confusion matrices")
        
        n_models = len(evaluation_results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (name, results) in enumerate(evaluation_results.items()):
            cm = confusion_matrix(y_test, results['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{name.replace("_", " ").title()}\nConfusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/confusion_matrices.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrices saved to {self.output_dir}/confusion_matrices.png")
    
    def save_models(self, evaluation_results):
        """Save all trained models"""
        logger.info("Saving trained models")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, results in evaluation_results.items():
            model_path = f"{self.models_dir}/{name}_{timestamp}.joblib"
            joblib.dump(results['model'], model_path)
            logger.info(f"Saved {name} to {model_path}")
        
        # Save scaler
        scaler_path = f"{self.models_dir}/scaler_{timestamp}.joblib"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
        
        # Save best model separately
        if self.best_model is not None:
            best_model_path = f"{self.models_dir}/best_model_{timestamp}.joblib"
            joblib.dump(self.best_model, best_model_path)
            logger.info(f"Saved best model ({self.best_model_name}) to {best_model_path}")
        
        # Save feature names
        feature_names_path = f"{self.models_dir}/feature_names_{timestamp}.txt"
        with open(feature_names_path, 'w') as f:
            for feature in self.feature_names:
                f.write(f"{feature}\n")
        logger.info(f"Saved feature names to {feature_names_path}")
    
    def generate_training_report(self, evaluation_results, comparison_df):
        """Generate comprehensive training report"""
        logger.info("Generating training report")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{self.output_dir}/training_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("MENTAL HEALTH RISK PREDICTION - MODEL TRAINING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Best Model: {self.best_model_name}\n\n")
            
            f.write("MODEL PERFORMANCE COMPARISON:\n")
            f.write("-" * 30 + "\n")
            f.write(comparison_df.to_string())
            f.write("\n\n")
            
            f.write("DETAILED MODEL RESULTS:\n")
            f.write("-" * 25 + "\n")
            for name, results in evaluation_results.items():
                f.write(f"\n{name.upper()}:\n")
                for metric, value in results['metrics'].items():
                    f.write(f"  {metric}: {value:.4f}\n")
        
        logger.info(f"Training report saved to {report_path}")
    
    def train_and_evaluate(self, data_path):
        """Main method to train and evaluate all models"""
        logger.info("Starting complete training and evaluation pipeline")
        
        # Load and prepare data
        df = self.load_data(data_path)
        X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test = self.prepare_data(df)
        
        # Define models
        self.define_models()
        
        # Train models
        training_results = self.train_models(X_train_scaled, y_train)
        
        # Evaluate models
        evaluation_results = self.evaluate_models(training_results, X_test_scaled, y_test)
        
        # Select best model
        best_name, best_results = self.select_best_model(evaluation_results)
        
        # Create visualizations
        comparison_df = self.create_model_comparison_plot(evaluation_results)
        self.create_confusion_matrices(evaluation_results, y_test)
        
        # Save models
        self.save_models(evaluation_results)
        
        # Generate report
        self.generate_training_report(evaluation_results, comparison_df)
        
        logger.info("Training and evaluation pipeline completed successfully!")
        
        return evaluation_results, best_name, best_results

if __name__ == "__main__":
    # Example usage
    trainer = MentalHealthModelTrainer()
    results, best_name, best_results = trainer.train_and_evaluate("data/processed/cleaned_data.csv")
    print("Model training completed successfully!") 