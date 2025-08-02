"""
Streamlit Web Application for Mental Health Risk Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set page config
st.set_page_config(
    page_title="Mental Health Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class MentalHealthPredictorApp:
    """
    Streamlit application for Mental Health Risk Prediction
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.explainer = None
        
    def load_model(self):
        """Load the trained model and related files"""
        try:
            # Find the latest model files with timestamps
            import glob
            
            model_files = glob.glob("models/best_model_*.joblib")
            scaler_files = glob.glob("models/scaler_*.joblib")
            feature_files = glob.glob("models/feature_names_*.txt")
            
            if not model_files or not scaler_files or not feature_files:
                st.error("Model files not found. Please ensure the model has been trained.")
                return False
            
            # Use the latest files (most recent timestamp)
            latest_model = max(model_files, key=os.path.getctime)
            latest_scaler = max(scaler_files, key=os.path.getctime)
            latest_features = max(feature_files, key=os.path.getctime)
            
            st.info(f"Loading model from: {latest_model}")
            
            self.model = joblib.load(latest_model)
            self.scaler = joblib.load(latest_scaler)
            
            with open(latest_features, 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            
            # Create SHAP explainer
            if hasattr(self.model, 'feature_importances_'):
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # For linear models, we'll create explainer when needed
                pass
            
            st.success(f"✅ Model loaded successfully! ({type(self.model).__name__})")
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def create_input_form(self):
        """Create the input form for user data"""
        st.markdown('<h1 class="main-header">🧠 Mental Health Risk Predictor</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem;">Predict mental health risk for remote workers using machine learning</p>', unsafe_allow_html=True)
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h2 class="sub-header">Personal Information</h2>', unsafe_allow_html=True)
            
            age = st.slider("Age", min_value=18, max_value=80, value=30, step=1)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
            st.markdown('<h2 class="sub-header">Work Environment</h2>', unsafe_allow_html=True)
            
            is_remote_worker = st.selectbox("Do you work remotely?", ["No", "Yes"])
            tech_company = st.selectbox("Do you work for a tech company?", ["No", "Yes"])
            
            company_size = st.selectbox("Company Size", [
                "1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"
            ])
            
            work_interfere = st.selectbox("How often does work interfere with your mental health?", [
                "Never", "Rarely", "Sometimes", "Often"
            ])
        
        with col2:
            st.markdown('<h2 class="sub-header">Mental Health History</h2>', unsafe_allow_html=True)
            
            family_history = st.selectbox("Do you have a family history of mental illness?", ["No", "Yes"])
            
            st.markdown('<h2 class="sub-header">Workplace Support</h2>', unsafe_allow_html=True)
            
            benefits = st.selectbox("Does your employer provide mental health benefits?", ["No", "Yes", "Don't know"])
            care_options = st.selectbox("Do you know the options for mental health care at your workplace?", ["No", "Yes", "Not sure"])
            wellness_program = st.selectbox("Does your employer offer wellness programs?", ["No", "Yes", "Don't know"])
            seek_help = st.selectbox("Would you seek help for a mental health issue?", ["No", "Yes", "Don't know"])
            anonymity = st.selectbox("Is anonymity protected if you seek mental health care?", ["No", "Yes", "Don't know"])
        
        # Additional features
        st.markdown('<h2 class="sub-header">Additional Factors</h2>', unsafe_allow_html=True)
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            leave = st.selectbox("How easy is it to take leave for mental health?", [
                "Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"
            ])
            
            mental_health_consequence = st.selectbox("Have you discussed mental health with your employer?", ["No", "Yes", "Maybe"])
            
        with col4:
            coworkers = st.selectbox("Would you discuss mental health with coworkers?", ["No", "Yes", "Some of them"])
            
            supervisor = st.selectbox("Would you discuss mental health with your supervisor?", ["No", "Yes", "Some of them"])
            
        with col5:
            mental_health_interview = st.selectbox("Would you bring up mental health in an interview?", ["No", "Yes", "Maybe"])
            
            obs_consequence = st.selectbox("Have you observed negative consequences for coworkers with mental health issues?", ["No", "Yes"])
        
        return {
            'Age': age,
            'Gender': gender,
            'is_remote_worker': 1 if is_remote_worker == "Yes" else 0,
            'tech_company': 1 if tech_company == "Yes" else 0,
            'no_employees': company_size,
            'work_interfere': work_interfere,
            'family_history': 1 if family_history == "Yes" else 0,
            'benefits': benefits,
            'care_options': care_options,
            'wellness_program': wellness_program,
            'seek_help': seek_help,
            'anonymity': anonymity,
            'leave': leave,
            'mental_health_consequence': mental_health_consequence,
            'coworkers': coworkers,
            'supervisor': supervisor,
            'mental_health_interview': mental_health_interview,
            'obs_consequence': 1 if obs_consequence == "Yes" else 0
        }
    
    def preprocess_input(self, user_input):
        """Preprocess user input for prediction"""
        # Create a DataFrame with the user input
        df = pd.DataFrame([user_input])
        
        # Encode categorical variables (simplified encoding)
        categorical_mappings = {
            'Gender': {'Male': 0, 'Female': 1, 'Other': 2},
            'no_employees': {
                '1-5': 1, '6-25': 2, '26-100': 3, '100-500': 4, 
                '500-1000': 5, 'More than 1000': 6
            },
            'work_interfere': {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3},
            'benefits': {'No': 0, 'Yes': 1, 'Don\'t know': 0.5, 'Not sure': 0.5},
            'care_options': {'No': 0, 'Yes': 1, 'Don\'t know': 0.5, 'Not sure': 0.5},
            'wellness_program': {'No': 0, 'Yes': 1, 'Don\'t know': 0.5},
            'seek_help': {'No': 0, 'Yes': 1, 'Don\'t know': 0.5},
            'anonymity': {'No': 0, 'Yes': 1, 'Don\'t know': 0.5},
            'leave': {
                'Very easy': 0, 'Somewhat easy': 1, 'Somewhat difficult': 2, 
                'Very difficult': 3, 'Don\'t know': 1.5
            },
            'mental_health_consequence': {'No': 0, 'Yes': 1, 'Maybe': 0.5},
            'coworkers': {'No': 0, 'Yes': 1, 'Some of them': 0.5},
            'supervisor': {'No': 0, 'Yes': 1, 'Some of them': 0.5},
            'mental_health_interview': {'No': 0, 'Yes': 1, 'Maybe': 0.5}
        }
        
        # Apply mappings
        for col, mapping in categorical_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        
        # Create engineered features
        df['work_stress_level'] = df['work_interfere']
        df['company_size_category'] = df['no_employees']
        
        # Calculate mental health support score
        support_features = ['benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity']
        support_score = df[support_features].sum(axis=1) / len(support_features)
        df['mental_health_support_score'] = support_score
        
        # Ensure all required features are present
        if self.feature_names:
            missing_features = set(self.feature_names) - set(df.columns)
            for feature in missing_features:
                df[feature] = 0  # Default value for missing features
        
        # Reorder columns to match model expectations
        if self.feature_names:
            df = df[self.feature_names]
        
        return df
    
    def make_prediction(self, processed_input):
        """Make prediction using the loaded model"""
        try:
            # Scale the input
            scaled_input = self.scaler.transform(processed_input)
            
            # Make prediction
            prediction = self.model.predict(scaled_input)[0]
            probability = self.model.predict_proba(scaled_input)[0]
            
            return prediction, probability
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None, None
    
    def create_shap_explanation(self, processed_input):
        """Create SHAP explanation for the prediction"""
        try:
            if self.explainer is None:
                return None
            
            scaled_input = self.scaler.transform(processed_input)
            shap_values = self.explainer.shap_values(scaled_input)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
            
            return shap_values[0]  # Return first (and only) sample
        except Exception as e:
            st.error(f"Error creating SHAP explanation: {str(e)}")
            return None
    
    def display_results(self, prediction, probability, shap_values, processed_input):
        """Display prediction results and explanations"""
        st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
        
        # Create metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_status = "High Risk" if prediction == 1 else "Low Risk"
            risk_color = "#ff6b6b" if prediction == 1 else "#4ecdc4"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: {risk_color};">Risk Level</h3>
                <h2 style="color: {risk_color};">{risk_status}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            confidence = probability[1] if prediction == 1 else probability[0]
            st.markdown(f"""
            <div class="metric-card">
                <h3>Confidence</h3>
                <h2>{confidence:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Model Used</h3>
                <h2>{type(self.model).__name__}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Display warning or info based on prediction
        if prediction == 1:
            st.markdown("""
            <div class="warning-box">
                <h4>⚠️ High Risk Detected</h4>
                <p>Based on your responses, you may be at higher risk for mental health challenges. 
                Consider reaching out to mental health professionals or your HR department for support.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <h4>✅ Low Risk Detected</h4>
                <p>Based on your responses, you appear to be at lower risk for mental health challenges. 
                Continue to prioritize your mental well-being and seek support if needed.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # SHAP Explanation
        if shap_values is not None:
            st.markdown('<h2 class="sub-header">Feature Importance Explanation</h2>', unsafe_allow_html=True)
            
            # Create SHAP explanation plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get feature names and values
            feature_names = self.feature_names if self.feature_names else [f"Feature_{i}" for i in range(len(shap_values))]
            
            # Create waterfall plot
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values,
                    base_values=self.explainer.expected_value,
                    data=processed_input.iloc[0].values,
                    feature_names=feature_names
                ),
                show=False
            )
            
            plt.title('SHAP Explanation - How Each Feature Contributed to the Prediction', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            st.pyplot(fig)
            plt.close()
            
            # Display top contributing factors
            st.markdown('<h3>Top Contributing Factors</h3>', unsafe_allow_html=True)
            
            # Get top features by absolute SHAP value
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'shap_value': shap_values
            }).sort_values('shap_value', key=abs, ascending=False)
            
            top_features = feature_importance.head(5)
            
            for _, row in top_features.iterrows():
                impact = "increased" if row['shap_value'] > 0 else "decreased"
                color = "red" if row['shap_value'] > 0 else "green"
                st.markdown(f"• **{row['feature']}**: {impact} risk by {abs(row['shap_value']):.3f}")
    
    def run(self):
        """Run the Streamlit application"""
        # Load model
        if not self.load_model():
            st.error("Failed to load model. Please ensure the model has been trained.")
            return
        
        # Create input form
        user_input = self.create_input_form()
        
        # Prediction button
        if st.button("🔮 Predict Mental Health Risk", type="primary", use_container_width=True):
            with st.spinner("Analyzing your responses..."):
                # Preprocess input
                processed_input = self.preprocess_input(user_input)
                
                # Make prediction
                prediction, probability = self.make_prediction(processed_input)
                
                if prediction is not None:
                    # Create SHAP explanation
                    shap_values = self.create_shap_explanation(processed_input)
                    
                    # Display results
                    self.display_results(prediction, probability, shap_values, processed_input)
        
        # Sidebar with information
        with st.sidebar:
            st.markdown("## 📊 About This Tool")
            st.markdown("""
            This mental health risk predictor uses machine learning to assess 
            the likelihood of mental health challenges based on workplace and 
            personal factors.
            
            **How it works:**
            1. Answer questions about your work environment
            2. Provide information about mental health support
            3. Get an AI-powered risk assessment
            4. Understand which factors contribute most to the prediction
            
            **Disclaimer:**
            This tool is for educational purposes only and should not replace 
            professional medical advice. If you're experiencing mental health 
            challenges, please seek help from qualified professionals.
            """)
            
            st.markdown("## 🆘 Resources")
            st.markdown("""
            **Crisis Support:**
            - National Suicide Prevention Lifeline: 988
            - Crisis Text Line: Text HOME to 741741
            
            **Mental Health Support:**
            - NAMI HelpLine: 1-800-950-NAMI
            - MentalHealth.gov
            
            **Workplace Support:**
            - Contact your HR department
            - Employee Assistance Program (EAP)
            """)

def main():
    """Main function to run the app"""
    app = MentalHealthPredictorApp()
    app.run()

if __name__ == "__main__":
    main() 