#!/usr/bin/env python3
"""
Deployment script for Mental Health Risk Predictor
Prepares the project for deployment on Streamlit Cloud
"""

import os
import shutil
import glob
import subprocess
import sys
from datetime import datetime

def create_deployment_files():
    """Create necessary files for deployment"""
    print("🚀 Preparing deployment files...")
    
    # Create deployment directory
    deploy_dir = "deployment"
    if os.path.exists(deploy_dir):
        shutil.rmtree(deploy_dir)
    os.makedirs(deploy_dir)
    
    # Copy essential files
    essential_files = [
        "app/streamlit_app.py",
        "requirements_deploy.txt",
        ".streamlit/config.toml",
        "packages.txt",
        "README.md"
    ]
    
    for file_path in essential_files:
        if os.path.exists(file_path):
            dest_path = os.path.join(deploy_dir, os.path.basename(file_path))
            shutil.copy2(file_path, dest_path)
            print(f"✅ Copied: {file_path}")
        else:
            print(f"⚠️  Warning: {file_path} not found")
    
    # Copy models directory
    if os.path.exists("models"):
        shutil.copytree("models", os.path.join(deploy_dir, "models"))
        print("✅ Copied: models/")
    
    # Copy src directory
    if os.path.exists("src"):
        shutil.copytree("src", os.path.join(deploy_dir, "src"))
        print("✅ Copied: src/")
    
    # Create a simple requirements.txt for deployment
    with open(os.path.join(deploy_dir, "requirements.txt"), "w") as f:
        f.write("""# Core dependencies for deployment
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=1.7.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
shap>=0.42.0
streamlit>=1.25.0
joblib>=1.3.0
pyyaml>=6.0.0
""")
    
    print("✅ Created: requirements.txt")
    
    # Create deployment README
    deploy_readme = """# Mental Health Risk Predictor - Deployment

This is the deployment version of the Mental Health Risk Prediction project.

## Quick Start

1. The app will automatically load the trained models
2. Fill out the form with your information
3. Get instant mental health risk predictions
4. View SHAP explanations for model interpretability

## Model Performance

- **Accuracy**: 92.03%
- **F1-Score**: 93.87%
- **Best Model**: XGBoost

## Features

- Real-time risk assessment
- Explainable AI with SHAP
- User-friendly interface
- Mental health resources

## Support

For questions or issues, please refer to the main project documentation.
"""
    
    with open(os.path.join(deploy_dir, "DEPLOYMENT_README.md"), "w") as f:
        f.write(deploy_readme)
    
    print("✅ Created: DEPLOYMENT_README.md")
    
    return deploy_dir

def create_github_workflow():
    """Create GitHub workflow for automatic deployment"""
    workflow_content = """name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements_deploy.txt
    
    - name: Run tests
      run: |
        python check_models.py
    
    - name: Deploy to Streamlit Cloud
      run: |
        echo "Deployment ready! Push to main branch to deploy on Streamlit Cloud."
"""
    
    workflow_dir = ".github/workflows"
    os.makedirs(workflow_dir, exist_ok=True)
    
    with open(os.path.join(workflow_dir, "deploy.yml"), "w") as f:
        f.write(workflow_content)
    
    print("✅ Created: .github/workflows/deploy.yml")

def create_dockerfile():
    """Create Dockerfile for alternative deployment"""
    dockerfile_content = """FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    software-properties-common \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_deploy.txt .
RUN pip install -r requirements_deploy.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8501

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the application
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    print("✅ Created: Dockerfile")

def create_heroku_files():
    """Create files for Heroku deployment"""
    # Procfile for Heroku
    procfile_content = """web: streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
"""
    
    with open("Procfile", "w") as f:
        f.write(procfile_content)
    
    print("✅ Created: Procfile")
    
    # runtime.txt for Heroku
    runtime_content = """python-3.10.12
"""
    
    with open("runtime.txt", "w") as f:
        f.write(runtime_content)
    
    print("✅ Created: runtime.txt")

def main():
    """Main deployment preparation function"""
    print("=" * 60)
    print("🧠 Mental Health Risk Predictor - Deployment Setup")
    print("=" * 60)
    
    # Create deployment files
    deploy_dir = create_deployment_files()
    
    # Create GitHub workflow
    create_github_workflow()
    
    # Create Dockerfile
    create_dockerfile()
    
    # Create Heroku files
    create_heroku_files()
    
    print("\n" + "=" * 60)
    print("🎉 Deployment files created successfully!")
    print("=" * 60)
    
    print("\n📁 Deployment directory created: deployment/")
    print("\n🚀 Deployment Options:")
    print("\n1. 🌐 Streamlit Cloud (Recommended):")
    print("   • Push your code to GitHub")
    print("   • Connect to https://share.streamlit.io/")
    print("   • Deploy automatically")
    
    print("\n2. 🐳 Docker:")
    print("   • Build: docker build -t mental-health-predictor .")
    print("   • Run: docker run -p 8501:8501 mental-health-predictor")
    
    print("\n3. ⚡ Heroku:")
    print("   • Install Heroku CLI")
    print("   • Run: heroku create your-app-name")
    print("   • Run: git push heroku main")
    
    print("\n4. 🐙 Render:")
    print("   • Connect your GitHub repo to Render")
    print("   • Deploy as a web service")
    
    print("\n📋 Next Steps:")
    print("1. Commit all changes to Git")
    print("2. Push to GitHub repository")
    print("3. Choose your deployment platform")
    print("4. Follow platform-specific instructions")
    
    print(f"\n⏰ Deployment prepared at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 