## README.md

# PS02 - AI-Powered Early Risk Prediction for Patient Health Conditions

Developed by Team **Algoriots**, this project is an AI-based healthcare system designed to analyze patient data and predict disease risks at an early stage. By processing multiple clinical datasets, the system identifies potential health issues before symptoms escalate into chronic diseases.

---

## 🚀 Overview

The system utilizes machine learning (ML) and deep learning (DNN) to provide risk assessments through a web-based interface. It processes specific disease datasets separately to ensure high accuracy and provides personalized preventive care recommendations.

### Key Features

* 
**Disease-Specific Modeling:** Separate training for various conditions such as liver disease, diabetes, and chronic kidney disease.


* 
**Risk Level Classification:** Predicts risk levels as **Low, Medium, or High** with associated probability scores.


* 
**Explainable AI:** Identifies key contributing factors for each prediction to assist in clinical decision-making.


* 
**Preventive Insights:** Delivers lifestyle modification suggestions and early medical consultation alerts.



---

## 🛠️ System Architecture

1. 
**Data Collection:** Datasets are sourced from the **UCI Machine Learning Repository**, Kaggle, and WHO reports.


2. 
**Preprocessing:** Includes data cleaning, handling missing values, normalization, and feature engineering for each specific disease.


3. 
**Model Training:** Employs algorithms like **XGBoost** and **Deep Neural Networks (DNN)**.


4. 
**Evaluation:** Models are validated using metrics such as Accuracy, Precision, Recall, F1-score, and ROC-AUC.


5. 
**User Interface:** A web interface for symptom input and risk visualization.



---

## 📂 Project Structure

* `envision_web_scrap.py`: A Python script used to programmatically fetch healthcare datasets (Liver Disorder, COVID-19 Surveillance, and Chronic Kidney Disease) from the UCI Repository.
* `/datasets`: (Folder) Directory where the fetched `.csv` files are stored for model training.
* `PS02- AI-Powered Early Risk Prediction...pdf`: Project documentation and conceptual framework.

---

## 💻 Setup & Usage

### Prerequisites

* Python 3.x
* `pandas`
* `ucimlrepo` library

### Data Acquisition

To download the necessary datasets into your local directory, run the scraping script:

```bash
python envision_web_scrap.py

```

This will generate the following files in your project folder:

* `liver_disorder.csv`
* `covid_19_surveillance.csv`
* `chronic_kidney_disease.csv`

---

## 🏥 Applications

* Early disease risk screening 


* Preventive healthcare planning 


* Clinical decision support for healthcare providers 


* Personalized health monitoring 

