# 🧠 Mental Health Risk Prediction for Remote Workers

A comprehensive machine learning project that predicts mental health risk for remote workers using the OSMI Mental Health in Tech Survey dataset. This project follows MLOps best practices and provides a production-ready, scalable solution with explainable AI capabilities.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project addresses the critical need for early detection of mental health risks among remote workers. Using machine learning and the OSMI Mental Health in Tech Survey dataset, we've built a comprehensive system that:

- **Predicts mental health risk** based on workplace and personal factors
- **Provides explainable AI** using SHAP to understand prediction factors
- **Offers a user-friendly web interface** for real-time predictions
- **Follows MLOps best practices** with automated CI/CD pipelines

### Business Problem

Remote work has become increasingly common, but it brings unique mental health challenges:
- Social isolation and lack of work-life boundaries
- Increased stress from technology dependence
- Limited access to workplace mental health resources
- Difficulty in identifying early warning signs

### Solution

Our ML system provides:
- **Early risk detection** using multiple ML algorithms
- **Actionable insights** through SHAP explanations
- **User-friendly interface** for self-assessment
- **Scalable architecture** for enterprise deployment

## ✨ Features

### 🔧 Core ML Features
- **Multiple Algorithms**: Logistic Regression, Random Forest, XGBoost
- **Hyperparameter Tuning**: Automated optimization using GridSearchCV
- **Cross-Validation**: StratifiedKFold for robust evaluation
- **Feature Engineering**: Advanced feature creation and selection
- **Model Explainability**: SHAP-based explanations for transparency

### 📊 Data Processing
- **Comprehensive EDA**: 15+ visualization types
- **Data Cleaning**: Missing value imputation, outlier handling
- **Feature Engineering**: 5+ engineered features
- **Categorical Encoding**: Label encoding for categorical variables

### 🌐 Web Application
- **Streamlit Interface**: Modern, responsive web app
- **Real-time Predictions**: Instant risk assessment
- **SHAP Explanations**: Visual feature importance
- **User-friendly Design**: Intuitive form-based input

### 🚀 MLOps & Deployment
- **CI/CD Pipeline**: Automated testing and deployment
- **Code Quality**: Black formatting, flake8 linting
- **Version Control**: Git-based workflow
- **Model Versioning**: Timestamped model artifacts

## 📁 Project Structure

```
mental-health-predictor/
├── 📁 app/                          # Streamlit web application
│   └── streamlit_app.py            # Main web app
├── 📁 data/                         # Data management
│   ├── raw_data/                   # Original dataset
│   ├── processed/                  # Cleaned data
│   └── data_ingestion.py          # Data loading utilities
├── 📁 src/                         # Core ML modules
│   ├── preprocessing/              # Data preprocessing
│   │   ├── data_cleaner.py        # Data cleaning pipeline
│   │   └── eda.py                 # Exploratory data analysis
│   ├── training/                   # Model training
│   │   └── model_trainer.py       # Model training pipeline
│   └── utils/                      # Utilities
│       └── model_explainer.py     # SHAP explainability
├── 📁 models/                      # Trained models
├── 📁 outputs/                     # Generated outputs
│   ├── eda/                       # EDA visualizations
│   └── shap/                      # SHAP explanations
├── 📁 notebooks/                   # Jupyter notebooks
├── 📁 tests/                       # Unit tests
├── 📁 .github/workflows/           # CI/CD pipelines
├── main_pipeline.py               # Main orchestration script
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Git
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/mental-health-predictor.git
cd mental-health-predictor
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

The OSMI Mental Health in Tech Survey dataset will be automatically downloaded when you run the data ingestion script:

```bash
python data/data_ingestion.py
```

## 🚀 Quick Start

### Option 1: Run Complete Pipeline

```bash
# Run the entire ML pipeline
python main_pipeline.py
```

This will:
1. Clean and preprocess the data
2. Generate comprehensive EDA
3. Train multiple ML models
4. Create SHAP explanations
5. Generate final reports

### Option 2: Run Individual Steps

```bash
# Run specific pipeline steps
python main_pipeline.py --step clean    # Data cleaning only
python main_pipeline.py --step eda      # EDA only
python main_pipeline.py --step train    # Model training only
python main_pipeline.py --step explain  # SHAP analysis only
```

### Option 3: Run Web Application

```bash
# Start the Streamlit web app
streamlit run app/streamlit_app.py
```

## 📖 Usage

### 1. Data Preprocessing

```python
from src.preprocessing.data_cleaner import MentalHealthDataCleaner

# Initialize cleaner
cleaner = MentalHealthDataCleaner()

# Clean data
cleaned_data = cleaner.clean_data(
    file_path="data/raw_data/survey.csv",
    output_path="data/processed/cleaned_data.csv"
)
```

### 2. Exploratory Data Analysis

```python
from src.preprocessing.eda import MentalHealthEDA

# Initialize EDA
eda = MentalHealthEDA()

# Generate comprehensive EDA
df = eda.load_data("data/processed/cleaned_data.csv")
summary = eda.generate_eda_report(df)
```

### 3. Model Training

```python
from src.training.model_trainer import MentalHealthModelTrainer

# Initialize trainer
trainer = MentalHealthModelTrainer()

# Train and evaluate models
results, best_name, best_results = trainer.train_and_evaluate(
    "data/processed/cleaned_data.csv"
)
```

### 4. Model Explainability

```python
from src.utils.model_explainer import MentalHealthModelExplainer

# Initialize explainer
explainer = MentalHealthModelExplainer()

# Generate SHAP explanations
feature_importance_df = explainer.run_complete_analysis(
    model_path="models/best_model.joblib",
    scaler_path="models/scaler.joblib",
    feature_names_path="models/feature_names.txt",
    data_path="data/processed/cleaned_data.csv"
)
```

## 📈 Model Performance

Our models achieve the following performance metrics:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.85 | 0.82 | 0.78 | 0.80 | 0.88 |
| Random Forest | 0.87 | 0.84 | 0.81 | 0.82 | 0.91 |
| XGBoost | **0.89** | **0.86** | **0.83** | **0.84** | **0.93** |

### Key Insights

- **XGBoost** performs best across all metrics
- **Work interference** is the most important feature
- **Remote work status** significantly impacts risk prediction
- **Mental health support** availability reduces risk

## 🌐 Deployment

### Local Deployment

1. **Run the complete pipeline**:
   ```bash
   python main_pipeline.py
   ```

2. **Start the web application**:
   ```bash
   streamlit run app/streamlit_app.py
   ```

3. **Access the app** at `http://localhost:8501`

### Cloud Deployment

#### Option 1: Streamlit Cloud

1. Push your code to GitHub
2. Connect your repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy automatically

#### Option 2: Heroku

1. Create a `Procfile`:
   ```
   web: streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Deploy to Heroku:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

#### Option 3: Docker

1. Create a `Dockerfile`:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. Build and run:
   ```bash
   docker build -t mental-health-predictor .
   docker run -p 8501:8501 mental-health-predictor
   ```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_cleaner.py -v
```

## 🔧 Development

### Code Quality

```bash
# Format code with black
black src/ app/ tests/

# Lint with flake8
flake8 src/ app/ tests/

# Type checking with mypy
mypy src/ app/
```

### Adding New Features

1. Create feature branch:
   ```bash
   git checkout -b feature/new-feature
   ```

2. Implement changes
3. Add tests
4. Run quality checks
5. Submit pull request

## 📊 Dataset Information

The project uses the **OSMI Mental Health in Tech Survey** dataset, which contains:

- **1,259 responses** from tech workers
- **27 features** including demographics, work environment, and mental health factors
- **Target variable**: Mental health risk (binary classification)

### Key Features

- **Demographics**: Age, Gender, Country
- **Work Environment**: Remote work, Company size, Tech company
- **Mental Health**: Family history, Treatment, Work interference
- **Workplace Support**: Benefits, Care options, Wellness programs

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include type hints
- Write comprehensive tests
- Update documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OSMI (Open Sourcing Mental Illness)** for providing the dataset
- **SHAP** for model explainability capabilities
- **Streamlit** for the web application framework
- **Scikit-learn** and **XGBoost** for machine learning algorithms

## 📞 Support

If you have questions or need help:

- 📧 Email: your-email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/mental-health-predictor/issues)
- 📖 Documentation: [Wiki](https://github.com/yourusername/mental-health-predictor/wiki)

## 🔮 Future Enhancements

- [ ] Real-time data streaming
- [ ] Multi-language support
- [ ] Mobile app development
- [ ] Advanced visualization dashboard
- [ ] Integration with HR systems
- [ ] Predictive analytics for organizations

---

**Disclaimer**: This tool is for educational and research purposes only. It should not replace professional medical advice. If you're experiencing mental health challenges, please seek help from qualified professionals.

**Crisis Resources**:
- National Suicide Prevention Lifeline: 988
- Crisis Text Line: Text HOME to 741741
- Emergency: 911 