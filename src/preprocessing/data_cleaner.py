"""
Data Cleaning and Preprocessing Module for Mental Health Risk Prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MentalHealthDataCleaner:
    """
    A comprehensive data cleaner for the OSMI Mental Health in Tech Survey dataset
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='most_frequent')
        self.categorical_columns = []
        self.numerical_columns = []
        
    def load_data(self, file_path):
        """Load the raw survey data"""
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def identify_column_types(self, df):
        """Identify categorical and numerical columns"""
        self.categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        self.numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove timestamp and comments from categorical columns
        if 'Timestamp' in self.categorical_columns:
            self.categorical_columns.remove('Timestamp')
        if 'comments' in self.categorical_columns:
            self.categorical_columns.remove('comments')
            
        logger.info(f"Categorical columns: {self.categorical_columns}")
        logger.info(f"Numerical columns: {self.numerical_columns}")
        
    def clean_gender_column(self, df):
        """Clean and standardize gender column"""
        logger.info("Cleaning gender column")
        
        # Create a mapping for gender standardization
        gender_mapping = {
            'Male': 'Male',
            'M': 'Male',
            'male': 'Male',
            'm': 'Male',
            'maile': 'Male',
            'Male-ish': 'Male',
            'Cis Male': 'Male',
            'Female': 'Female',
            'F': 'Female',
            'female': 'Female',
            'f': 'Female',
            'Cis Female': 'Female',
            'Trans-female': 'Transgender',
            'something kinda male?': 'Other',
            'Male-ish': 'Other',
            'Guy (-ish) ^_^': 'Other',
            'Enby': 'Other',
            'non-binary': 'Other',
            'Nah': 'Other',
            'All': 'Other',
            'ostensibly male, unsure what that really means': 'Other'
        }
        
        df['Gender'] = df['Gender'].map(gender_mapping).fillna('Other')
        return df
    
    def clean_age_column(self, df):
        """Clean age column by handling outliers and missing values"""
        logger.info("Cleaning age column")
        
        # Convert to numeric, coercing errors to NaN
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        
        # Remove extreme outliers (ages < 15 or > 80)
        df = df[(df['Age'] >= 15) & (df['Age'] <= 80)]
        
        # Fill missing ages with median
        median_age = df['Age'].median()
        df['Age'] = df['Age'].fillna(median_age)
        
        return df
    
    def create_target_variable(self, df):
        """Create binary target variable for mental health risk"""
        logger.info("Creating target variable")
        
        # Define high-risk conditions based on survey responses
        risk_conditions = (
            (df['treatment'] == 'Yes') |  # Currently seeking treatment
            (df['work_interfere'] == 'Often') |  # Work often interferes with mental health
            (df['mental_health_consequence'] == 'Yes') |  # Mental health consequences at work
            (df['family_history'] == 'Yes') & (df['work_interfere'].isin(['Sometimes', 'Often']))  # Family history + work interference
        )
        
        df['mental_health_risk'] = risk_conditions.astype(int)
        
        # Log target distribution
        risk_distribution = df['mental_health_risk'].value_counts()
        logger.info(f"Target distribution: {risk_distribution.to_dict()}")
        
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        logger.info("Handling missing values")
        
        # For categorical columns, fill with 'Unknown'
        for col in self.categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # For numerical columns, use median imputation
        for col in self.numerical_columns:
            if col in df.columns and col != 'mental_health_risk':
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        return df
    
    def encode_categorical_variables(self, df):
        """Encode categorical variables using Label Encoding"""
        logger.info("Encoding categorical variables")
        
        for col in self.categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                
        return df
    
    def create_feature_engineering(self, df):
        """Create new engineered features"""
        logger.info("Creating engineered features")
        
        # Binary flag for remote work
        df['is_remote_worker'] = (df['remote_work'] == 'Yes').astype(int)
        
        # Work stress level based on work interference
        work_stress_mapping = {
            'Never': 0,
            'Rarely': 1,
            'Sometimes': 2,
            'Often': 3
        }
        df['work_stress_level'] = df['work_interfere'].map(work_stress_mapping).fillna(1)
        
        # Company size category
        company_size_mapping = {
            '1-5': 1,
            '6-25': 2,
            '26-100': 3,
            '100-500': 4,
            '500-1000': 5,
            'More than 1000': 6
        }
        df['company_size_category'] = df['no_employees'].map(company_size_mapping).fillna(3)
        
        # Mental health support score (higher = more support)
        support_features = ['benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity']
        support_mapping = {'Yes': 1, 'No': 0, 'Don\'t know': 0.5, 'Not sure': 0.5}
        
        support_score = 0
        for feature in support_features:
            if feature in df.columns:
                support_score += df[feature].map(support_mapping).fillna(0)
        
        df['mental_health_support_score'] = support_score / len(support_features)
        
        return df
    
    def select_features(self, df):
        """Select relevant features for modeling"""
        logger.info("Selecting features for modeling")
        
        # Define features to include in the model
        feature_columns = [
            'Age', 'Gender', 'family_history', 'work_interfere', 'no_employees',
            'remote_work', 'tech_company', 'benefits', 'care_options', 'wellness_program',
            'seek_help', 'anonymity', 'leave', 'mental_health_consequence',
            'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview',
            'phys_health_interview', 'mental_vs_physical', 'obs_consequence',
            'is_remote_worker', 'work_stress_level', 'company_size_category',
            'mental_health_support_score'
        ]
        
        # Filter to only include columns that exist in the dataset
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Add target variable
        final_columns = available_features + ['mental_health_risk']
        
        df_clean = df[final_columns].copy()
        
        logger.info(f"Selected {len(available_features)} features for modeling")
        logger.info(f"Final dataset shape: {df_clean.shape}")
        
        return df_clean
    
    def clean_data(self, file_path, output_path=None):
        """Main method to clean the entire dataset"""
        logger.info("Starting data cleaning process")
        
        # Load data
        df = self.load_data(file_path)
        
        # Identify column types
        self.identify_column_types(df)
        
        # Clean specific columns
        df = self.clean_gender_column(df)
        df = self.clean_age_column(df)
        
        # Create target variable
        df = self.create_target_variable(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Create engineered features
        df = self.create_feature_engineering(df)
        
        # Encode categorical variables
        df = self.encode_categorical_variables(df)
        
        # Select final features
        df_clean = self.select_features(df)
        
        # Save cleaned data
        if output_path:
            df_clean.to_csv(output_path, index=False)
            logger.info(f"Cleaned data saved to {output_path}")
        
        return df_clean

if __name__ == "__main__":
    # Example usage
    cleaner = MentalHealthDataCleaner()
    cleaned_data = cleaner.clean_data(
        file_path="data/raw_data/survey.csv",
        output_path="data/processed/cleaned_data.csv"
    )
    print("Data cleaning completed successfully!") 