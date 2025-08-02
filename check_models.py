#!/usr/bin/env python3
"""
Script to check if model files are accessible and can be loaded
"""

import os
import glob
import joblib
import sys

def check_model_files():
    """Check if model files exist and can be loaded"""
    print("🔍 Checking model files...")
    
    # Check for model files
    model_files = glob.glob("models/best_model_*.joblib")
    scaler_files = glob.glob("models/scaler_*.joblib")
    feature_files = glob.glob("models/feature_names_*.txt")
    
    print(f"Found {len(model_files)} model files")
    print(f"Found {len(scaler_files)} scaler files")
    print(f"Found {len(feature_files)} feature files")
    
    if not model_files or not scaler_files or not feature_files:
        print("❌ Missing model files!")
        return False
    
    # Get latest files
    latest_model = max(model_files, key=os.path.getctime)
    latest_scaler = max(scaler_files, key=os.path.getctime)
    latest_features = max(feature_files, key=os.path.getctime)
    
    print(f"Latest model: {latest_model}")
    print(f"Latest scaler: {latest_scaler}")
    print(f"Latest features: {latest_features}")
    
    try:
        # Try to load the model
        print("\n🔄 Loading model...")
        model = joblib.load(latest_model)
        print(f"✅ Model loaded successfully: {type(model).__name__}")
        
        # Try to load the scaler
        print("🔄 Loading scaler...")
        scaler = joblib.load(latest_scaler)
        print(f"✅ Scaler loaded successfully: {type(scaler).__name__}")
        
        # Try to load feature names
        print("🔄 Loading feature names...")
        with open(latest_features, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        print(f"✅ Feature names loaded: {len(feature_names)} features")
        
        print("\n🎉 All model files are accessible and can be loaded!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model files: {e}")
        return False

def test_prediction():
    """Test a simple prediction"""
    print("\n🧪 Testing prediction...")
    
    try:
        import glob
        import joblib
        import numpy as np
        
        # Load latest model
        model_files = glob.glob("models/best_model_*.joblib")
        scaler_files = glob.glob("models/scaler_*.joblib")
        feature_files = glob.glob("models/feature_names_*.txt")
        
        latest_model = max(model_files, key=os.path.getctime)
        latest_scaler = max(scaler_files, key=os.path.getctime)
        latest_features = max(feature_files, key=os.path.getctime)
        
        model = joblib.load(latest_model)
        scaler = joblib.load(latest_scaler)
        
        with open(latest_features, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        # Create a sample input (all zeros for testing)
        sample_input = np.zeros((1, len(feature_names)))
        
        # Scale the input
        scaled_input = scaler.transform(sample_input)
        
        # Make prediction
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0]
        
        print(f"✅ Prediction test successful!")
        print(f"   Prediction: {prediction}")
        print(f"   Probability: {probability}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("🧠 Mental Health Risk Predictor - Model Check")
    print("=" * 50)
    
    # Check model files
    files_ok = check_model_files()
    
    if files_ok:
        # Test prediction
        prediction_ok = test_prediction()
        
        if prediction_ok:
            print("\n🎉 All tests passed! The model is ready to use.")
            print("\n🌐 You can now run the Streamlit app:")
            print("   streamlit run app/streamlit_app.py")
        else:
            print("\n❌ Prediction test failed.")
            sys.exit(1)
    else:
        print("\n❌ Model file check failed.")
        sys.exit(1) 