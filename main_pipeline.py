"""
Main Pipeline for Mental Health Risk Prediction Project
Orchestrates the complete ML workflow from data to deployment
"""

import os
import sys
import logging
from datetime import datetime
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing.data_cleaner import MentalHealthDataCleaner
from preprocessing.eda import MentalHealthEDA
from training.model_trainer import MentalHealthModelTrainer
from utils.model_explainer import MentalHealthModelExplainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MentalHealthPipeline:
    """
    Main pipeline orchestrator for Mental Health Risk Prediction
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.start_time = datetime.now()
        
        # Create necessary directories
        self.create_directories()
        
        logger.info("Mental Health Risk Prediction Pipeline initialized")
    
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            'data/processed',
            'models',
            'outputs/eda',
            'outputs/shap',
            'logs'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def run_data_cleaning(self):
        """Run data cleaning and preprocessing"""
        logger.info("=" * 50)
        logger.info("STEP 1: DATA CLEANING AND PREPROCESSING")
        logger.info("=" * 50)
        
        try:
            cleaner = MentalHealthDataCleaner()
            cleaned_data = cleaner.clean_data(
                file_path="data/raw_data/survey.csv",
                output_path="data/processed/cleaned_data.csv"
            )
            
            logger.info(f"Data cleaning completed. Cleaned data shape: {cleaned_data.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            return False
    
    def run_eda(self):
        """Run exploratory data analysis"""
        logger.info("=" * 50)
        logger.info("STEP 2: EXPLORATORY DATA ANALYSIS")
        logger.info("=" * 50)
        
        try:
            eda = MentalHealthEDA()
            df = eda.load_data("data/processed/cleaned_data.csv")
            summary = eda.generate_eda_report(df)
            
            logger.info("EDA completed successfully")
            logger.info(f"Summary: {summary}")
            return True
            
        except Exception as e:
            logger.error(f"EDA failed: {str(e)}")
            return False
    
    def run_model_training(self):
        """Run model training and evaluation"""
        logger.info("=" * 50)
        logger.info("STEP 3: MODEL TRAINING AND EVALUATION")
        logger.info("=" * 50)
        
        try:
            trainer = MentalHealthModelTrainer()
            results, best_name, best_results = trainer.train_and_evaluate(
                "data/processed/cleaned_data.csv"
            )
            
            logger.info(f"Model training completed. Best model: {best_name}")
            logger.info(f"Best model F1-score: {best_results['metrics']['f1_score']:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return False
    
    def run_model_explanation(self):
        """Run SHAP model explanation"""
        logger.info("=" * 50)
        logger.info("STEP 4: MODEL EXPLANABILITY (SHAP)")
        logger.info("=" * 50)
        
        try:
            # Find the latest model files
            import glob
            model_files = glob.glob("models/best_model_*.joblib")
            scaler_files = glob.glob("models/scaler_*.joblib")
            feature_files = glob.glob("models/feature_names_*.txt")
            
            if not model_files or not scaler_files or not feature_files:
                logger.warning("Model files not found. Skipping SHAP analysis.")
                return True
            
            # Use the latest files
            latest_model = max(model_files, key=os.path.getctime)
            latest_scaler = max(scaler_files, key=os.path.getctime)
            latest_features = max(feature_files, key=os.path.getctime)
            
            explainer = MentalHealthModelExplainer()
            feature_importance_df = explainer.run_complete_analysis(
                model_path=latest_model,
                scaler_path=latest_scaler,
                feature_names_path=latest_features,
                data_path="data/processed/cleaned_data.csv"
            )
            
            logger.info("SHAP analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"SHAP analysis failed: {str(e)}")
            return False
    
    def generate_final_report(self):
        """Generate final pipeline report"""
        logger.info("=" * 50)
        logger.info("STEP 5: GENERATING FINAL REPORT")
        logger.info("=" * 50)
        
        try:
            end_time = datetime.now()
            duration = end_time - self.start_time
            
            report_content = f"""
MENTAL HEALTH RISK PREDICTION - PIPELINE EXECUTION REPORT
{'=' * 60}

Pipeline Execution Summary:
- Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
- End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
- Total Duration: {duration}

Pipeline Steps Completed:
1. ✅ Data Cleaning and Preprocessing
2. ✅ Exploratory Data Analysis
3. ✅ Model Training and Evaluation
4. ✅ Model Explainability (SHAP)

Output Files Generated:
- Cleaned Data: data/processed/cleaned_data.csv
- EDA Visualizations: outputs/eda/
- Trained Models: models/
- SHAP Explanations: outputs/shap/

Next Steps:
1. Review the generated visualizations in outputs/eda/
2. Check model performance in the training report
3. Examine SHAP explanations for model interpretability
4. Deploy the best model using the Streamlit app

To run the Streamlit app:
streamlit run app/streamlit_app.py

Pipeline completed successfully!
"""
            
            # Save report
            with open("pipeline_execution_report.txt", "w") as f:
                f.write(report_content)
            
            logger.info("Final report generated: pipeline_execution_report.txt")
            print(report_content)
            
            return True
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            return False
    
    def run_complete_pipeline(self):
        """Run the complete pipeline"""
        logger.info("Starting Mental Health Risk Prediction Pipeline")
        logger.info(f"Pipeline started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Data Cleaning
        if not self.run_data_cleaning():
            logger.error("Pipeline failed at data cleaning step")
            return False
        
        # Step 2: EDA
        if not self.run_eda():
            logger.error("Pipeline failed at EDA step")
            return False
        
        # Step 3: Model Training
        if not self.run_model_training():
            logger.error("Pipeline failed at model training step")
            return False
        
        # Step 4: Model Explanation
        if not self.run_model_explanation():
            logger.error("Pipeline failed at model explanation step")
            return False
        
        # Step 5: Generate Final Report
        if not self.generate_final_report():
            logger.error("Pipeline failed at report generation step")
            return False
        
        logger.info("Pipeline completed successfully!")
        return True

def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(description="Mental Health Risk Prediction Pipeline")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--step", type=str, choices=["clean", "eda", "train", "explain", "all"], 
                       default="all", help="Pipeline step to run")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MentalHealthPipeline()
    
    # Run pipeline based on arguments
    if args.step == "all":
        success = pipeline.run_complete_pipeline()
    elif args.step == "clean":
        success = pipeline.run_data_cleaning()
    elif args.step == "eda":
        success = pipeline.run_eda()
    elif args.step == "train":
        success = pipeline.run_model_training()
    elif args.step == "explain":
        success = pipeline.run_model_explanation()
    
    if success:
        logger.info("Pipeline execution completed successfully!")
        sys.exit(0)
    else:
        logger.error("Pipeline execution failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 