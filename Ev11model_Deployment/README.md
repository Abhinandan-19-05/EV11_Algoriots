# 🏥 Multi-Disease Risk Predictor

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-square&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)

A comprehensive Streamlit web application for predicting three major health conditions using machine learning models. This application provides risk assessment for Liver Disease, Diabetes, and Obesity categories with an integrated AI medical chatbot.

![App Screenshot](https://via.placeholder.com/800x400.png?text=Multi-Disease+Risk+Predictor+Screenshot)

---

## ✨ Features

### 🩺 **Liver Disease Prediction**
- **8 Clinical Biomarkers:**
  - Age, Total Bilirubin, Direct Bilirubin
  - Alkaline Phosphotase (ALT), Alamine Aminotransferase (AST)
  - Albumin, Albumin/Globulin Ratio
- Automatic feature name translation
- Real-time normal range indicators
- Probability-based risk assessment

### 🩺 **Diabetes Risk Assessment**
- **11 Health Factors:**
  - Blood pressure, Cholesterol, BMI
  - Heart disease history, Physical activity
  - General health status, Mobility issues
  - Demographics (Age, Education, Income)
- Interactive input forms with tabs
- Real-time risk score calculation
- Comprehensive probability breakdown

### ⚖️ **Obesity Category Prediction**
- **6 Body Metrics:**
  - Age, Gender, Height, Weight
  - BMI (auto-calculated)
  - Physical Activity Level
- Automatic BMI calculation
- 6-category classification (Normal to Extremely Obese)
- Color-coded BMI indicators

### 🤖 **MediAssist AI Chatbot**
- **Integrated Medical Assistant:**
  - Direct access to AI medical chatbot
  - 24/7 health advice availability
  - One-click access from sidebar
  - Opens in new tab for seamless experience

---

## 🚀 Live Demo

> **🔗 Live Application:** [Click here to access the live demo](https://envision-9yhgtix3kqygzi7anyswgu.streamlit.app/)  
> *Note: Replace with your actual deployment link*

---

## 🏗️ Architecture

```
├── models/
│   ├── Cleaned_Liver_disease.pkl        # Liver disease pipeline
│   ├── Cleaned_diabetes_disease.pkl     # Diabetes pipeline
│   └── obesity_data_pipline.pkl         # Obesity pipeline
├── app.py                               # Main Streamlit application
├── requirements.txt                     # Python dependencies
└── README.md                            # This documentation
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/multi-disease-predictor.git
cd multi-disease-predictor
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Model Files
Place your pre-trained pipeline files in the root directory:
- `Cleaned_Liver_disease.pkl`
- `Cleaned_diabetes_disease.pkl`
- `obesity_data_pipline.pkl`

### Step 4: Run the Application
```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

---

## 📊 Model Details

### Liver Disease Model
- **Algorithm:** Random Forest Classifier
- **Features:** 8 clinical biomarkers
- **Pipeline:** Scaler + Model
- **Output:** Binary classification (Disease/No Disease)

### Diabetes Model
- **Algorithm:** Random Forest Classifier
- **Features:** 11 health factors
- **Pipeline:** Scaler + Model
- **Output:** Binary classification (Diabetes/No Diabetes)

### Obesity Model
- **Algorithm:** Random Forest Classifier
- **Features:** 6 body metrics
- **Pipeline:** Scaler + Model
- **Output:** 6-class classification (Normal to Extremely Obese)

---

## 🎨 User Interface

### Home Page
- Three gradient model cards with feature lists
- Clear "How to Use" instructions
- Technical information section
- Prominent chatbot announcement

### Prediction Pages
- Clean, organized input forms
- Real-time validation and calculations
- Color-coded results display
- Professional metric cards

### Sidebar Navigation
- Easy model selection
- Real-time model status indicators
- Integrated chatbot button
- Medical disclaimer

---

## 🔧 Technical Implementation

### Feature Name Translation
```python
def translate_liver_features(user_inputs):
    # Handles different feature naming conventions
    # Converts 'Alkaline Phosphatase' → 'Alkaline Phosphotase'
    # Adds missing features like 'Total Bilirubin'
    # Maps 'Alamine Aminotransferase' correctly
```

### Model Pipeline Structure
```python
# Each model uses a Scikit-learn Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),   # Already fitted
    ('model', RandomForestClassifier())  # Already trained
])
```

### Chatbot Integration
```python
# Opens MediAssist AI in new tab
chatbot_url = "https://mediassist-ai-v6q9djfkdz73epitt6rrh9.streamlit.app/"
js = f'window.open("{chatbot_url}", "_blank")'
```

---

## 📁 File Structure

```
multi-disease-predictor/
│
├── app.py                              # Main application
├── requirements.txt                    # Python dependencies
├── README.md                           # This documentation
│
├── models/                             # Pre-trained models
│   ├── Cleaned_Liver_disease.pkl
│   ├── Cleaned_diabetes_disease.pkl
│   └── obesity_data_pipline.pkl
│
├── screenshots/                        # Application screenshots
│   ├── home_page.png
│   ├── liver_prediction.png
│   └── diabetes_prediction.png
│
└── docs/                               # Additional documentation
    ├── model_details.md
    └── deployment_guide.md
```

---

## ⚠️ Important Notes

### Feature Requirements
- **Liver Model:** Requires 8 specific features with exact naming
- **Diabetes Model:** Uses 11 features from CDC BRFSS dataset
- **Obesity Model:** Requires 6 body metrics with specific encoding

### Medical Disclaimer
> **⚠️ IMPORTANT:** This application provides risk assessment only and is not a substitute for professional medical advice. Always consult qualified healthcare providers for diagnosis and treatment decisions.

### Model Compatibility
- Models must be saved as `.pkl` files using `joblib`
- Feature names must match training data exactly
- Pipelines should include scaler and model components

---

## 🔗 External Integration

### MediAssist AI Chatbot
- **URL:** `https://mediassist-ai-v6q9djfkdz73epitt6rrh9.streamlit.app/`
- **Access:** One-click button in sidebar
- **Purpose:** Provides additional medical advice and explanations

---

## 🚀 Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Cloud Deployment Options
1. **Streamlit Community Cloud** (Recommended)
   ```bash
   streamlit login
   streamlit deploy app.py
   ```

2. **Heroku**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

3. **AWS/GCP/Azure**
   - Containerize with Docker
   - Deploy to cloud container service

### Environment Variables
```bash
# Optional: For production deployment
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

---

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Error**
   ```
   Solution: Ensure .pkl files are in correct directory
   ```

2. **Feature Name Mismatch**
   ```
   Solution: Check model training feature names
   ```

3. **Port Already in Use**
   ```
   Solution: Change port: streamlit run app.py --server.port 8502
   ```

4. **Missing Dependencies**
   ```
   Solution: pip install -r requirements.txt
   ```

---

## 📈 Future Enhancements

### Planned Features
- [ ] User authentication system
- [ ] Prediction history database
- [ ] Export results as PDF
- [ ] Mobile app version
- [ ] Additional disease models
- [ ] Multi-language support

### Technical Improvements
- [ ] Model version management
- [ ] Automated testing
- [ ] Performance optimization
- [ ] Enhanced visualization
- [ ] API endpoints

---

## 👥 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset Providers:** Medical research institutions
- **Streamlit Team:** For the amazing web app framework
- **Scikit-learn Team:** For machine learning tools
- **Contributors:** All who helped improve this project

---

## 📞 Support

For questions, issues, or suggestions:
- **GitHub Issues:** [Create an issue](https://github.com/yourusername/multi-disease-predictor/issues)
- **Email:** your.email@example.com
- **Documentation:** See `/docs` folder for detailed guides

---

<div align="center">
  
### 🚀 Ready to Get Started?

**Replace this section with your actual deployment link:**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Click_Here-FF4B4B?style=for-square&logo=streamlit&logoColor=white)](YOUR_DEPLOYMENT_LINK_HERE)

*Click above to try the live application!*

</div>

---

*Last Updated: December 2024*  
*Version: 1.0.0*
