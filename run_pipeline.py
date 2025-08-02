#!/usr/bin/env python3
"""
Simple script to run the Mental Health Risk Prediction pipeline
"""

import sys
import os
import subprocess
import time
from datetime import datetime

def print_banner():
    """Print a nice banner for the pipeline"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║  🧠 Mental Health Risk Prediction Pipeline                   ║
    ║                                                              ║
    ║  A comprehensive ML project for predicting mental health     ║
    ║  risk among remote workers using the OSMI dataset           ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Check if required files exist
    required_files = [
        "requirements.txt",
        "data/raw_data/survey.csv",
        "src/preprocessing/data_cleaner.py",
        "src/preprocessing/eda.py",
        "src/training/model_trainer.py",
        "src/utils/model_explainer.py"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Required file not found: {file_path}")
            return False
    
    print("✅ All required files found")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print(f"Error output: {e.stderr}")
        return False

def run_pipeline():
    """Run the main pipeline"""
    print("🚀 Starting Mental Health Risk Prediction Pipeline...")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Run the main pipeline
        result = subprocess.run([sys.executable, "main_pipeline.py"], 
                              check=True, capture_output=False)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("=" * 60)
        print(f"✅ Pipeline completed successfully!")
        print(f"⏱️  Total duration: {duration:.2f} seconds")
        print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline failed with error code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        return False

def show_next_steps():
    """Show next steps after successful pipeline execution"""
    print("\n" + "=" * 60)
    print("🎉 Pipeline completed! Here's what you can do next:")
    print("=" * 60)
    
    print("\n📊 View Results:")
    print("   • EDA visualizations: outputs/eda/")
    print("   • Model performance: outputs/training_report_*.txt")
    print("   • SHAP explanations: outputs/shap/")
    print("   • Trained models: models/")
    
    print("\n🌐 Run Web Application:")
    print("   streamlit run app/streamlit_app.py")
    print("   Then open: http://localhost:8501")
    
    print("\n🧪 Run Tests:")
    print("   pytest tests/ -v")
    
    print("\n📖 View Documentation:")
    print("   • README.md - Complete project documentation")
    print("   • pipeline_execution_report.txt - Pipeline summary")
    
    print("\n🔧 Development:")
    print("   • Format code: black src/ app/ tests/")
    print("   • Lint code: flake8 src/ app/ tests/")
    print("   • Run specific steps: python main_pipeline.py --step <step_name>")

def main():
    """Main function"""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed. Please fix the issues above.")
        sys.exit(1)
    
    # Ask user if they want to install dependencies
    print("\n📦 Dependencies check:")
    try:
        import pandas
        import numpy
        import sklearn
        import xgboost
        import streamlit
        import shap
        print("✅ All required packages are already installed")
    except ImportError:
        print("⚠️  Some required packages are missing")
        response = input("Do you want to install missing dependencies? (y/n): ")
        if response.lower() in ['y', 'yes']:
            if not install_dependencies():
                print("\n❌ Failed to install dependencies. Please install them manually.")
                sys.exit(1)
        else:
            print("❌ Cannot proceed without required dependencies.")
            sys.exit(1)
    
    # Run pipeline
    print("\n" + "=" * 60)
    response = input("Ready to run the pipeline? (y/n): ")
    if response.lower() not in ['y', 'yes']:
        print("Pipeline execution cancelled.")
        sys.exit(0)
    
    success = run_pipeline()
    
    if success:
        show_next_steps()
    else:
        print("\n❌ Pipeline failed. Please check the error messages above.")
        print("💡 You can also run individual steps:")
        print("   python main_pipeline.py --step clean")
        print("   python main_pipeline.py --step eda")
        print("   python main_pipeline.py --step train")
        print("   python main_pipeline.py --step explain")
        sys.exit(1)

if __name__ == "__main__":
    main() 