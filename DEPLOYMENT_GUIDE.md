# 🚀 Deployment Guide - Mental Health Risk Predictor

This guide will help you deploy your Mental Health Risk Prediction project to various free hosting platforms.

## 📋 Prerequisites

Before deploying, ensure you have:
- ✅ Project working locally
- ✅ GitHub account
- ✅ All deployment files created (run `python deploy_to_streamlit.py`)

## 🌐 Option 1: Streamlit Cloud (Recommended)

**Best for**: Quick deployment, free hosting, automatic updates

### Step 1: Push to GitHub
```bash
# Initialize git if not already done
git init
git add .
git commit -m "Initial commit: Mental Health Risk Predictor"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/mental-health-predictor.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [https://share.streamlit.io/](https://share.streamlit.io/)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `mental-health-predictor`
5. Set the main file path: `app/streamlit_app.py`
6. Click "Deploy!"

**Your app will be live at**: `https://your-app-name.streamlit.app`

---

## 🐳 Option 2: Docker + Render

**Best for**: More control, custom domain, scalability

### Step 1: Create Render Account
1. Go to [https://render.com/](https://render.com/)
2. Sign up with GitHub
3. Create a new account

### Step 2: Deploy on Render
1. Click "New +" → "Web Service"
2. Connect your GitHub repository
3. Configure:
   - **Name**: `mental-health-predictor`
   - **Environment**: `Docker`
   - **Branch**: `main`
   - **Root Directory**: Leave empty
   - **Build Command**: `docker build -t mental-health-predictor .`
   - **Start Command**: `docker run -p $PORT:8501 mental-health-predictor`

4. Click "Create Web Service"

**Your app will be live at**: `https://your-app-name.onrender.com`

---

## ⚡ Option 3: Heroku

**Best for**: Traditional hosting, PostgreSQL integration

### Step 1: Install Heroku CLI
```bash
# Download from: https://devcenter.heroku.com/articles/heroku-cli
```

### Step 2: Deploy to Heroku
```bash
# Login to Heroku
heroku login

# Create Heroku app
heroku create your-mental-health-app

# Add buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Open the app
heroku open
```

**Your app will be live at**: `https://your-mental-health-app.herokuapp.com`

---

## 🐙 Option 4: Railway

**Best for**: Simple deployment, good free tier

### Step 1: Deploy on Railway
1. Go to [https://railway.app/](https://railway.app/)
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway will automatically detect it's a Python app
6. Deploy!

**Your app will be live at**: `https://your-app-name.railway.app`

---

## 🔧 Option 5: Local Docker Deployment

**Best for**: Testing, development, self-hosting

### Step 1: Build and Run
```bash
# Build the Docker image
docker build -t mental-health-predictor .

# Run the container
docker run -p 8501:8501 mental-health-predictor
```

**Your app will be live at**: `http://localhost:8501`

---

## 📊 Deployment Checklist

Before deploying, ensure:

### ✅ Code Quality
- [ ] All tests pass: `python check_models.py`
- [ ] Code formatted: `black src/ app/ tests/`
- [ ] No linting errors: `flake8 src/ app/ tests/`

### ✅ Files Present
- [ ] `app/streamlit_app.py` - Main application
- [ ] `requirements.txt` - Dependencies
- [ ] `models/` - Trained models
- [ ] `src/` - Source code
- [ ] `.streamlit/config.toml` - Streamlit config
- [ ] `Dockerfile` - Docker configuration
- [ ] `Procfile` - Heroku configuration

### ✅ Model Files
- [ ] `models/best_model_*.joblib` - Best trained model
- [ ] `models/scaler_*.joblib` - Feature scaler
- [ ] `models/feature_names_*.txt` - Feature names

---

## 🚨 Troubleshooting

### Common Issues

#### 1. Model Files Not Found
**Error**: "Model files not found. Please ensure the model has been trained."

**Solution**:
```bash
# Run the pipeline to generate models
python main_pipeline.py

# Check if models exist
python check_models.py
```

#### 2. Dependencies Issues
**Error**: "ModuleNotFoundError: No module named 'shap'"

**Solution**:
```bash
# Update requirements.txt with exact versions
pip freeze > requirements.txt
```

#### 3. Memory Issues
**Error**: "MemoryError" or "OutOfMemoryError"

**Solution**:
- Reduce model complexity
- Use smaller dataset for training
- Optimize SHAP calculations

#### 4. Port Issues
**Error**: "Port already in use"

**Solution**:
```bash
# Kill existing processes
taskkill /f /im streamlit.exe  # Windows
pkill -f streamlit  # Linux/Mac

# Or use different port
streamlit run app/streamlit_app.py --server.port 8502
```

---

## 🔒 Security Considerations

### Environment Variables
For production, set these environment variables:
```bash
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
```

### Data Privacy
- ✅ No sensitive data in code
- ✅ Models don't contain personal information
- ✅ Input data is not stored
- ✅ HTTPS enabled on production

---

## 📈 Monitoring and Maintenance

### Health Checks
- Monitor app uptime
- Check model performance
- Review user feedback
- Update dependencies regularly

### Performance Optimization
- Cache model loading
- Optimize SHAP calculations
- Use CDN for static assets
- Implement request rate limiting

---

## 🎯 Recommended Deployment Strategy

### For Beginners: Streamlit Cloud
1. **Easiest setup**
2. **Free hosting**
3. **Automatic deployments**
4. **Good for MVPs**

### For Production: Render + Docker
1. **More control**
2. **Custom domains**
3. **Better scalability**
4. **Professional setup**

---

## 📞 Support

If you encounter issues:

1. **Check logs**: Look at deployment platform logs
2. **Test locally**: Ensure it works locally first
3. **Update dependencies**: Keep packages updated
4. **Community help**: Stack Overflow, GitHub Issues

---

## 🎉 Success!

Once deployed, your Mental Health Risk Predictor will be:
- ✅ **Publicly accessible**
- ✅ **Real-time predictions**
- ✅ **SHAP explanations**
- ✅ **User-friendly interface**
- ✅ **Mental health resources**

**Congratulations on deploying your ML project! 🚀** 