"""
Unit tests for the data cleaner module
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from src.preprocessing.data_cleaner import MentalHealthDataCleaner

class TestMentalHealthDataCleaner:
    """Test cases for MentalHealthDataCleaner class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        data = {
            'Timestamp': ['2014-08-27 11:29:31', '2014-08-27 11:29:37'],
            'Age': [37, 44],
            'Gender': ['Female', 'M'],
            'Country': ['United States', 'United States'],
            'state': ['IL', 'IN'],
            'self_employed': [np.nan, np.nan],
            'family_history': ['No', 'No'],
            'treatment': ['Yes', 'No'],
            'work_interfere': ['Often', 'Rarely'],
            'no_employees': ['6-25', 'More than 1000'],
            'remote_work': ['No', 'No'],
            'tech_company': ['Yes', 'No'],
            'benefits': ['Yes', "Don't know"],
            'care_options': ['Not sure', 'No'],
            'wellness_program': ['No', 'No'],
            'seek_help': ['Yes', "Don't know"],
            'anonymity': ['Yes', "Don't know"],
            'leave': ['Somewhat easy', "Don't know"],
            'mental_health_consequence': ['No', 'Maybe'],
            'phys_health_consequence': ['No', 'No'],
            'coworkers': ['Some of them', 'No'],
            'supervisor': ['Yes', 'No'],
            'mental_health_interview': ['No', 'No'],
            'phys_health_interview': ['Maybe', 'Maybe'],
            'mental_vs_physical': ['Yes', 'No'],
            'obs_consequence': ['No', 'No'],
            'comments': [np.nan, np.nan]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def cleaner(self):
        """Create a MentalHealthDataCleaner instance"""
        return MentalHealthDataCleaner()
    
    def test_cleaner_initialization(self, cleaner):
        """Test cleaner initialization"""
        assert cleaner.label_encoders == {}
        assert cleaner.categorical_columns == []
        assert cleaner.numerical_columns == []
    
    def test_load_data(self, cleaner, sample_data):
        """Test data loading functionality"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            loaded_data = cleaner.load_data(temp_file)
            assert isinstance(loaded_data, pd.DataFrame)
            assert len(loaded_data) == 2
            assert len(loaded_data.columns) == 27
        finally:
            os.unlink(temp_file)
    
    def test_identify_column_types(self, cleaner, sample_data):
        """Test column type identification"""
        cleaner.identify_column_types(sample_data)
        
        # Check that categorical columns are identified
        assert 'Gender' in cleaner.categorical_columns
        assert 'Country' in cleaner.categorical_columns
        assert 'Timestamp' not in cleaner.categorical_columns  # Should be removed
        assert 'comments' not in cleaner.categorical_columns   # Should be removed
        
        # Check that numerical columns are identified
        assert 'Age' in cleaner.numerical_columns
    
    def test_clean_gender_column(self, cleaner, sample_data):
        """Test gender column cleaning"""
        cleaned_data = cleaner.clean_gender_column(sample_data)
        
        # Check that gender values are standardized
        assert cleaned_data['Gender'].iloc[0] == 'Female'
        assert cleaned_data['Gender'].iloc[1] == 'Male'  # 'M' should become 'Male'
        
        # Check that all values are in the expected set
        expected_genders = {'Male', 'Female', 'Other'}
        assert all(gender in expected_genders for gender in cleaned_data['Gender'])
    
    def test_clean_age_column(self, cleaner, sample_data):
        """Test age column cleaning"""
        # Add some problematic age values
        sample_data.loc[2] = [None, 999, 'Male', 'US', None, None, 'No', 'No', 'Never', '6-25', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Easy', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'No', None]
        sample_data.loc[3] = [None, 5, 'Female', 'US', None, None, 'No', 'No', 'Never', '6-25', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Easy', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'No', None]
        
        cleaned_data = cleaner.clean_age_column(sample_data)
        
        # Check that outliers are removed
        assert len(cleaned_data) < len(sample_data)  # Outliers should be removed
        
        # Check that ages are within reasonable range
        assert all(15 <= age <= 80 for age in cleaned_data['Age'] if pd.notna(age))
    
    def test_create_target_variable(self, cleaner, sample_data):
        """Test target variable creation"""
        result_data = cleaner.create_target_variable(sample_data)
        
        # Check that target variable is created
        assert 'mental_health_risk' in result_data.columns
        
        # Check that target is binary
        assert all(risk in [0, 1] for risk in result_data['mental_health_risk'])
        
        # Check that high-risk conditions create target=1
        # First row has treatment='Yes' and work_interfere='Often', so should be high risk
        assert result_data['mental_health_risk'].iloc[0] == 1
    
    def test_handle_missing_values(self, cleaner, sample_data):
        """Test missing value handling"""
        # Add some missing values
        sample_data.loc[2] = [None, np.nan, None, 'US', None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
        
        cleaner.identify_column_types(sample_data)
        cleaned_data = cleaner.handle_missing_values(sample_data)
        
        # Check that no missing values remain in categorical columns
        for col in cleaner.categorical_columns:
            if col in cleaned_data.columns:
                assert not cleaned_data[col].isna().any()
        
        # Check that numerical columns have reasonable imputed values
        for col in cleaner.numerical_columns:
            if col in cleaned_data.columns and col != 'mental_health_risk':
                assert not cleaned_data[col].isna().any()
    
    def test_encode_categorical_variables(self, cleaner, sample_data):
        """Test categorical variable encoding"""
        cleaner.identify_column_types(sample_data)
        encoded_data = cleaner.encode_categorical_variables(sample_data)
        
        # Check that categorical columns are encoded
        for col in cleaner.categorical_columns:
            if col in encoded_data.columns:
                # Check that values are numeric
                assert encoded_data[col].dtype in ['int64', 'int32']
                
                # Check that label encoders are stored
                assert col in cleaner.label_encoders
    
    def test_create_feature_engineering(self, cleaner, sample_data):
        """Test feature engineering"""
        engineered_data = cleaner.create_feature_engineering(sample_data)
        
        # Check that new features are created
        assert 'is_remote_worker' in engineered_data.columns
        assert 'work_stress_level' in engineered_data.columns
        assert 'company_size_category' in engineered_data.columns
        assert 'mental_health_support_score' in engineered_data.columns
        
        # Check that engineered features have correct values
        assert engineered_data['is_remote_worker'].iloc[0] == 0  # 'No' should become 0
        assert 0 <= engineered_data['work_stress_level'].iloc[0] <= 3
        assert 1 <= engineered_data['company_size_category'].iloc[0] <= 6
        assert 0 <= engineered_data['mental_health_support_score'].iloc[0] <= 1
    
    def test_select_features(self, cleaner, sample_data):
        """Test feature selection"""
        # First create engineered features
        sample_data = cleaner.create_feature_engineering(sample_data)
        sample_data = cleaner.create_target_variable(sample_data)
        
        selected_data = cleaner.select_features(sample_data)
        
        # Check that target variable is included
        assert 'mental_health_risk' in selected_data.columns
        
        # Check that important features are included
        important_features = ['Age', 'Gender', 'family_history', 'work_interfere']
        for feature in important_features:
            if feature in sample_data.columns:
                assert feature in selected_data.columns
    
    def test_complete_cleaning_pipeline(self, cleaner, sample_data):
        """Test the complete cleaning pipeline"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            cleaned_data = cleaner.clean_data(temp_file)
            
            # Check that data is cleaned and processed
            assert isinstance(cleaned_data, pd.DataFrame)
            assert 'mental_health_risk' in cleaned_data.columns
            assert len(cleaned_data) > 0
            
            # Check that no missing values in important columns
            important_cols = ['Age', 'mental_health_risk']
            for col in important_cols:
                if col in cleaned_data.columns:
                    assert not cleaned_data[col].isna().any()
                    
        finally:
            os.unlink(temp_file)
    
    def test_error_handling(self, cleaner):
        """Test error handling for invalid inputs"""
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            cleaner.load_data("non_existent_file.csv")
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises(Exception):
            cleaner.identify_column_types(empty_df)

if __name__ == "__main__":
    pytest.main([__file__]) 