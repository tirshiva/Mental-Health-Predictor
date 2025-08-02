#!/usr/bin/env python3
"""
Quick Deployment Script for Mental Health Risk Predictor
Guides you through the deployment process step by step
"""

import os
import sys
import subprocess
import webbrowser
from datetime import datetime

def print_banner():
    """Print deployment banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║  🚀 Mental Health Risk Predictor - Quick Deployment         ║
    ║                                                              ║
    ║  Choose your deployment platform and follow the steps       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("🔍 Checking prerequisites...")
    
    # Check if git is installed
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        print("✅ Git is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Git is not installed. Please install Git first.")
        return False
    
    # Check if models exist
    if not os.path.exists("models"):
        print("❌ Models directory not found. Please run the pipeline first.")
        return False
    
    model_files = [f for f in os.listdir("models") if f.endswith(".joblib")]
    if not model_files:
        print("❌ No model files found. Please run the pipeline first.")
        return False
    
    print(f"✅ Found {len(model_files)} model files")
    
    # Check if deployment files exist
    required_files = [
        "app/streamlit_app.py",
        "requirements.txt",
        "Dockerfile",
        "Procfile"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Required file not found: {file_path}")
            return False
    
    print("✅ All required files found")
    return True

def streamlit_cloud_deployment():
    """Guide through Streamlit Cloud deployment"""
    print("\n" + "=" * 50)
    print("🌐 Streamlit Cloud Deployment")
    print("=" * 50)
    
    print("\n📋 Steps to deploy on Streamlit Cloud:")
    print("1. Push your code to GitHub")
    print("2. Visit https://share.streamlit.io/")
    print("3. Sign in with GitHub")
    print("4. Click 'New app'")
    print("5. Select your repository")
    print("6. Set main file: app/streamlit_app.py")
    print("7. Click 'Deploy!'")
    
    response = input("\nWould you like to open Streamlit Cloud in your browser? (y/n): ")
    if response.lower() in ['y', 'yes']:
        webbrowser.open("https://share.streamlit.io/")
    
    print("\n✅ Streamlit Cloud deployment guide completed!")

def render_deployment():
    """Guide through Render deployment"""
    print("\n" + "=" * 50)
    print("🐳 Render Deployment")
    print("=" * 50)
    
    print("\n📋 Steps to deploy on Render:")
    print("1. Visit https://render.com/")
    print("2. Sign up with GitHub")
    print("3. Click 'New +' → 'Web Service'")
    print("4. Connect your GitHub repository")
    print("5. Configure as Docker service")
    print("6. Deploy!")
    
    response = input("\nWould you like to open Render in your browser? (y/n): ")
    if response.lower() in ['y', 'yes']:
        webbrowser.open("https://render.com/")
    
    print("\n✅ Render deployment guide completed!")

def heroku_deployment():
    """Guide through Heroku deployment"""
    print("\n" + "=" * 50)
    print("⚡ Heroku Deployment")
    print("=" * 50)
    
    print("\n📋 Steps to deploy on Heroku:")
    print("1. Install Heroku CLI")
    print("2. Run: heroku login")
    print("3. Run: heroku create your-app-name")
    print("4. Run: git push heroku main")
    print("5. Run: heroku open")
    
    print("\n🔗 Heroku CLI download: https://devcenter.heroku.com/articles/heroku-cli")
    
    response = input("\nWould you like to download Heroku CLI? (y/n): ")
    if response.lower() in ['y', 'yes']:
        webbrowser.open("https://devcenter.heroku.com/articles/heroku-cli")
    
    print("\n✅ Heroku deployment guide completed!")

def railway_deployment():
    """Guide through Railway deployment"""
    print("\n" + "=" * 50)
    print("🐙 Railway Deployment")
    print("=" * 50)
    
    print("\n📋 Steps to deploy on Railway:")
    print("1. Visit https://railway.app/")
    print("2. Sign in with GitHub")
    print("3. Click 'New Project'")
    print("4. Select 'Deploy from GitHub repo'")
    print("5. Select your repository")
    print("6. Deploy!")
    
    response = input("\nWould you like to open Railway in your browser? (y/n): ")
    if response.lower() in ['y', 'yes']:
        webbrowser.open("https://railway.app/")
    
    print("\n✅ Railway deployment guide completed!")

def docker_local_deployment():
    """Guide through local Docker deployment"""
    print("\n" + "=" * 50)
    print("🐳 Local Docker Deployment")
    print("=" * 50)
    
    print("\n📋 Steps for local Docker deployment:")
    print("1. Install Docker Desktop")
    print("2. Build: docker build -t mental-health-predictor .")
    print("3. Run: docker run -p 8501:8501 mental-health-predictor")
    print("4. Open: http://localhost:8501")
    
    response = input("\nWould you like to download Docker Desktop? (y/n): ")
    if response.lower() in ['y', 'yes']:
        webbrowser.open("https://www.docker.com/products/docker-desktop/")
    
    print("\n✅ Docker deployment guide completed!")

def git_setup():
    """Help with Git setup"""
    print("\n" + "=" * 50)
    print("📚 Git Setup Help")
    print("=" * 50)
    
    print("\nIf you haven't set up Git yet, here are the commands:")
    print("\n1. Initialize Git repository:")
    print("   git init")
    
    print("\n2. Add all files:")
    print("   git add .")
    
    print("\n3. Make initial commit:")
    print("   git commit -m 'Initial commit: Mental Health Risk Predictor'")
    
    print("\n4. Create main branch:")
    print("   git branch -M main")
    
    print("\n5. Add remote repository (replace with your GitHub repo):")
    print("   git remote add origin https://github.com/YOUR_USERNAME/mental-health-predictor.git")
    
    print("\n6. Push to GitHub:")
    print("   git push -u origin main")
    
    response = input("\nWould you like to create a GitHub repository? (y/n): ")
    if response.lower() in ['y', 'yes']:
        webbrowser.open("https://github.com/new")
    
    print("\n✅ Git setup guide completed!")

def main():
    """Main deployment guide function"""
    print_banner()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        sys.exit(1)
    
    print("\n✅ All prerequisites met! Ready for deployment.")
    
    # Show deployment options
    print("\n🚀 Choose your deployment platform:")
    print("1. 🌐 Streamlit Cloud (Recommended - Easiest)")
    print("2. 🐳 Render (Docker-based)")
    print("3. ⚡ Heroku (Traditional)")
    print("4. 🐙 Railway (Simple)")
    print("5. 🔧 Local Docker")
    print("6. 📚 Git Setup Help")
    print("7. ❌ Exit")
    
    while True:
        choice = input("\nEnter your choice (1-7): ")
        
        if choice == "1":
            streamlit_cloud_deployment()
        elif choice == "2":
            render_deployment()
        elif choice == "3":
            heroku_deployment()
        elif choice == "4":
            railway_deployment()
        elif choice == "5":
            docker_local_deployment()
        elif choice == "6":
            git_setup()
        elif choice == "7":
            print("\n👋 Goodbye! Good luck with your deployment!")
            break
        else:
            print("❌ Invalid choice. Please enter a number between 1-7.")
        
        # Ask if user wants to try another option
        if choice != "7":
            another = input("\nWould you like to try another deployment option? (y/n): ")
            if another.lower() not in ['y', 'yes']:
                print("\n👋 Goodbye! Good luck with your deployment!")
                break
    
    print(f"\n⏰ Deployment guide completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 