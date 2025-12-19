import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Multi-Disease Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Sub-header styling */
    .sub-header {
        font-size: 2rem;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 700;
        border-bottom: 3px solid #3B82F6;
        padding-bottom: 0.5rem;
    }
    
    /* Model cards styling */
    .model-card {
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        min-height: 300px;
    }
    
    .model-card h3 {
        margin-top: 0;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .model-card ul {
        margin-left: 1rem;
        margin-bottom: 0;
    }
    
    .model-card li {
        margin-bottom: 0.5rem;
    }
    
    /* Specific card colors */
    .liver-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .diabetes-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .obesity-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        text-align: center;
        border-top: 4px solid #3B82F6;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #3B82F6, #1D4ED8);
        color: white;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4);
    }
    
    /* Chatbot button specific styling */
    .chatbot-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
    }
    
    .chatbot-button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* History cards */
    .history-card {
        background: #F8FAFC;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #3B82F6;
    }
    
    /* Section divider */
    .section-divider {
        height: 3px;
        background: linear-gradient(90deg, #3B82F6, transparent);
        margin: 2rem 0;
    }
    
    /* Feature note */
    .feature-note {
        font-size: 0.85rem;
        color: #6B7280;
        font-style: italic;
        margin-top: 0.25rem;
    }
    
    /* BMI indicators */
    .bmi-normal { color: #059669; font-weight: bold; }
    .bmi-overweight { color: #D97706; font-weight: bold; }
    .bmi-obese { color: #DC2626; font-weight: bold; }
    .bmi-underweight { color: #3B82F6; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD PIPELINES ====================
@st.cache_resource
def load_liver_pipeline():
    """Load liver pipeline"""
    try:
        pipeline = joblib.load('Cleaned_Liver_disease.pkl')
        return pipeline
    except:
        return None

@st.cache_resource
def load_diabetes_pipeline():
    """Load diabetes pipeline"""
    try:
        pipeline = joblib.load('Cleaned_diabetes_disease.pkl')
        return pipeline
    except:
        return None

@st.cache_resource
def load_obesity_pipeline():
    """Load obesity pipeline"""
    try:
        pipeline = joblib.load('obesity_data_pipline.pkl')
        return pipeline
    except:
        return None

# ==================== CHATBOT FUNCTION ====================
def open_chatbot():
    """Open the MediAssist AI chatbot in a new tab"""
    chatbot_url = "https://mediassist-ai-v6q9djfkdz73epitt6rrh9.streamlit.app/"
    # Create JavaScript to open in new tab
    js = f"""
    <script>
        window.open("{chatbot_url}", "_blank");
    </script>
    """
    # Also show a message
    st.success("Opening MediAssist AI Chatbot in new tab...")
    # Execute the JavaScript
    st.components.v1.html(js, height=0)

# ==================== HOME PAGE ====================
def home_page():
    """Home page with model information"""
    st.markdown('<div class="main-header">🏥 Multi-Disease Risk Predictor</div>', unsafe_allow_html=True)
    
    st.markdown("### 🌟 Three Health Assessment Models")
    
    # Create three columns for model cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="model-card liver-card">
            <h3>🩺 Liver Disease</h3>
            <p><strong>8 Features Required:</strong></p>
            <ul>
                <li>Age</li>
                <li>Total Bilirubin</li>
                <li>Direct Bilirubin</li>
                <li>Alkaline Phosphotase</li>
                <li>Alamine Aminotransferase (ALT)</li>
                <li>Aspartate Aminotransferase (AST)</li>
                <li>Albumin</li>
                <li>Albumin/Globulin Ratio</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="model-card diabetes-card">
            <h3>🩺 Diabetes</h3>
            <p><strong>11 Features:</strong></p>
            <ul>
                <li>High BP, High Cholesterol</li>
                <li>BMI, Heart Disease</li>
                <li>Physical Activity</li>
                <li>General Health</li>
                <li>Walking Difficulty</li>
                <li>Age, Education, Income</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="model-card obesity-card">
            <h3>⚖️ Obesity</h3>
            <p><strong>6 Features:</strong></p>
            <ul>
                <li>Age, Gender</li>
                <li>Height, Weight</li>
                <li>BMI (auto-calculated)</li>
                <li>Physical Activity Level</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Chatbot announcement
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 2rem; border-radius: 10px; text-align: center;">
        <h3 style="margin-top: 0;">🤖 Need Medical Advice?</h3>
        <p style="font-size: 1.1rem;">Click the <strong>MediAssist AI Chatbot</strong> button in the sidebar 
        to get personalized medical advice and answers to your health questions!</p>
        <p style="font-size: 0.9rem; opacity: 0.9;">
        <em>Powered by advanced AI - Available 24/7</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==================== LIVER DISEASE PREDICTION ====================
def liver_disease_prediction():
    """Liver Disease Prediction with corrected features"""
    st.markdown('<div class="sub-header">🩺 Liver Disease Risk Assessment</div>', unsafe_allow_html=True)
    
    pipeline = load_liver_pipeline()
    if pipeline is None:
        st.error("Liver model not loaded.")
        return
    
    # Input form for ALL required features
    st.markdown("### 📝 Enter Complete Liver Function Test Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Basic Information")
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=45, step=1)
        
        st.markdown("#### Bilirubin Tests")
        total_bilirubin = st.number_input("Total Bilirubin (mg/dL)", 
                                         min_value=0.0, max_value=30.0, value=0.8, step=0.1,
                                         help="Normal: 0.2-1.2 mg/dL")
        direct_bilirubin = st.number_input("Direct Bilirubin (mg/dL)",
                                          min_value=0.0, max_value=20.0, value=0.5, step=0.1,
                                          help="Normal: 0.1-0.3 mg/dL")
    
    with col2:
        st.markdown("#### Liver Enzymes")
        
        alkaline_phosphatase = st.number_input("Alkaline Phosphatase (IU/L)", 
                                              min_value=20, max_value=1400, value=187, step=1)
        st.markdown('<p class="feature-note">Model expects: Alkaline Phosphotase</p>', unsafe_allow_html=True)
        
        alamine_aminotransferase = st.number_input("Alamine Aminotransferase (ALT) (IU/L)", 
                                                  min_value=5, max_value=500, value=30, step=1)
        st.markdown('<p class="feature-note">Normal: 7-56 IU/L (ALT)</p>', unsafe_allow_html=True)
        
        aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase (AST) (IU/L)",
                                                    min_value=5, max_value=500, value=25, step=1)
        st.markdown('<p class="feature-note">Normal: 10-40 IU/L (AST)</p>', unsafe_allow_html=True)
    
    with col3:
        st.markdown("#### Protein Levels")
        albumin = st.number_input("Albumin (g/dL)", min_value=1.0, max_value=6.0, value=4.5, step=0.1)
        albumin_globulin_ratio = st.number_input("Albumin/Globulin Ratio", 
                                                min_value=0.1, max_value=3.0, value=1.2, step=0.1)
    
    # Prediction button
    if st.button("📊 Predict Liver Disease Risk", type="primary"):
        with st.spinner("Analyzing biomarkers..."):
            try:
                # Try different feature name combinations
                feature_variations = [
                    {
                        'Age': age,
                        'Total Bilirubin': total_bilirubin,
                        'Direct Bilirubin': direct_bilirubin,
                        'Alkaline Phosphotase': alkaline_phosphatase,
                        'Alamine Aminotransferase': alamine_aminotransferase,
                        'Aspartate Aminotransferase': aspartate_aminotransferase,
                        'Albumin': albumin,
                        'Albumin and Globulin Ratio': albumin_globulin_ratio
                    },
                    {
                        'Age': age,
                        'Total_Bilirubin': total_bilirubin,
                        'Direct_Bilirubin': direct_bilirubin,
                        'Alkaline_Phosphotase': alkaline_phosphatase,
                        'Alamine_Aminotransferase': alamine_aminotransferase,
                        'Aspartate_Aminotransferase': aspartate_aminotransferase,
                        'Albumin': albumin,
                        'Albumin_and_Globulin_Ratio': albumin_globulin_ratio
                    }
                ]
                
                prediction_made = False
                for features in feature_variations:
                    try:
                        input_df = pd.DataFrame([features])
                        prediction = pipeline.predict(input_df)[0]
                        probabilities = pipeline.predict_proba(input_df)[0]
                        prediction_made = True
                        break
                    except:
                        continue
                
                if not prediction_made:
                    st.error("Feature matching failed.")
                    return
                
                # Display results
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                st.markdown("## 📈 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    diagnosis = "Liver Disease" if prediction == 1 else "No Liver Disease"
                    st.metric("Diagnosis", diagnosis)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    disease_prob = probabilities[1] * 100
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Probability", f"{disease_prob:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    confidence = max(probabilities) * 100
                    st.metric("Confidence", f"{confidence:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

# ==================== DIABETES PREDICTION ====================
def diabetes_prediction():
    """Diabetes Prediction"""
    st.markdown('<div class="sub-header">🩺 Diabetes Risk Assessment</div>', unsafe_allow_html=True)
    
    pipeline = load_diabetes_pipeline()
    if pipeline is None:
        st.error("Diabetes model not loaded.")
        return
    
    # Input form
    st.markdown("### 📝 Enter Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        high_bp = st.selectbox("High Blood Pressure", [0, 1],
                              format_func=lambda x: "Yes" if x == 1 else "No")
        high_chol = st.selectbox("High Cholesterol", [0, 1],
                                format_func=lambda x: "Yes" if x == 1 else "No")
        bmi = st.slider("Body Mass Index (BMI)", 12.0, 98.0, 25.0, 0.1)
        heart_disease = st.selectbox("Heart Disease or Attack", [0, 1],
                                    format_func=lambda x: "Yes" if x == 1 else "No")
        phys_activity = st.selectbox("Physical Activity", [0, 1],
                                    format_func=lambda x: "Yes" if x == 1 else "No")
    
    with col2:
        gen_hlth = st.select_slider("General Health", options=[1, 2, 3, 4, 5], value=3,
                                   format_func=lambda x: {1: "Excellent", 2: "Very Good", 
                                                          3: "Good", 4: "Fair", 5: "Poor"}[x])
        phys_hlth = st.slider("Poor Physical Health Days", 0, 30, 0)
        diff_walk = st.selectbox("Difficulty Walking", [0, 1],
                                format_func=lambda x: "Yes" if x == 1 else "No")
        age = st.slider("Age Category", 1, 13, 7)
        education = st.slider("Education Level", 1, 6, 4)
        income = st.slider("Income Level", 1, 8, 5)
    
    # Prediction button
    if st.button("📊 Predict Diabetes Risk", type="primary"):
        with st.spinner("Analyzing risk factors..."):
            try:
                input_data = {
                    'HighBP': high_bp,
                    'HighChol': high_chol,
                    'BMI': bmi,
                    'HeartDiseaseorAttack': heart_disease,
                    'PhysActivity': phys_activity,
                    'GenHlth': gen_hlth,
                    'PhysHlth': phys_hlth,
                    'DiffWalk': diff_walk,
                    'Age': age,
                    'Education': education,
                    'Income': income
                }
                
                input_df = pd.DataFrame([input_data])
                prediction = pipeline.predict(input_df)[0]
                probabilities = pipeline.predict_proba(input_df)[0]
                
                # Display results
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                st.markdown("## 📈 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    diagnosis = "Diabetes" if prediction == 1 else "No Diabetes"
                    st.metric("Diagnosis", diagnosis)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    diabetes_prob = probabilities[1] * 100
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Probability", f"{diabetes_prob:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    confidence = max(probabilities) * 100
                    st.metric("Confidence", f"{confidence:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

# ==================== OBESITY PREDICTION ====================
def obesity_prediction():
    """Obesity Category Prediction"""
    st.markdown('<div class="sub-header">⚖️ Obesity Category Prediction</div>', unsafe_allow_html=True)
    
    pipeline = load_obesity_pipeline()
    if pipeline is None:
        st.error("Obesity model not loaded.")
        return
    
    # Input form
    st.markdown("### 📝 Enter Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=30, step=1)
        gender = st.selectbox("Gender", [0, 1], 
                            format_func=lambda x: "Male" if x == 0 else "Female")
        height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.1)
    
    with col2:
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.1)
        
        # Calculate BMI
        if height > 0:
            height_m = height / 100
            bmi_calculated = weight / (height_m * height_m)
            st.info(f"**Calculated BMI:** {bmi_calculated:.1f}")
    
    with col3:
        bmi = st.number_input("BMI", value=float(bmi_calculated) if 'bmi_calculated' in locals() else 25.0, 
                             min_value=10.0, max_value=60.0, step=0.1)
        
        physical_activity = st.selectbox("Physical Activity Level", [1, 2, 3, 4, 5],
                                       format_func=lambda x: {
                                           1: "Sedentary",
                                           2: "Light",
                                           3: "Moderate",
                                           4: "Active",
                                           5: "Very Active"
                                       }[x])
    
    # Prediction button
    if st.button("📊 Predict Obesity Category", type="primary"):
        with st.spinner("Analyzing body metrics..."):
            try:
                input_data = {
                    'Age': age,
                    'Gender': gender,
                    'Height': height,
                    'Weight': weight,
                    'BMI': bmi,
                    'PhysicalActivityLevel': physical_activity
                }
                
                input_df = pd.DataFrame([input_data])
                prediction = pipeline.predict(input_df)[0]
                
                # Map prediction to category
                category_mapping = {
                    0: "Normal weight",
                    1: "Overweight",
                    2: "Obese",
                    3: "Underweight",
                    4: "Extremely obese",
                    5: "Insufficient weight"
                }
                
                predicted_category = category_mapping.get(prediction, f"Category {prediction}")
                
                # Display results
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                st.markdown("## 📈 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Predicted Category", predicted_category)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Category Code", prediction)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    # Try to get probabilities
                    try:
                        probabilities = pipeline.predict_proba(input_df)[0]
                        top_prob = max(probabilities) * 100
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Confidence", f"{top_prob:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    except:
                        pass
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

# ==================== MAIN APP ====================
def main():
    """Main application"""
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2917/2917633.png", width=100)
        st.markdown("## 🔍 Navigation")
        
        app_mode = st.radio(
            "Select Model:",
            ["🏠 Home", "🩺 Liver Disease", "🩺 Diabetes", "⚖️ Obesity"],
            index=0
        )
        
        st.markdown("---")
        
        # Model status
        st.markdown("### 📊 Model Status")
        liver_model = load_liver_pipeline()
        diabetes_model = load_diabetes_pipeline()
        obesity_model = load_obesity_pipeline()
        
        if liver_model:
            st.success("✅ Liver (8 features)")
        else:
            st.error("❌ Liver model")
        
        if diabetes_model:
            st.success("✅ Diabetes (11 features)")
        else:
            st.error("❌ Diabetes model")
        
        if obesity_model:
            st.success("✅ Obesity (6 features)")
        else:
            st.error("❌ Obesity model")
        
        st.markdown("---")
        
        # ==================== CHATBOT BUTTON ====================
        st.markdown("### 🤖 Medical Assistant")
        
        # Create a container for the chatbot button with custom styling
        chatbot_container = st.container()
        with chatbot_container:
            # Add custom CSS for this specific button
            st.markdown("""
            <style>
            div[data-testid="stButton"] > button[kind="secondary"] {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: white !important;
                border: none !important;
                font-weight: bold !important;
                padding: 1rem !important;
                border-radius: 8px !important;
                margin: 1rem 0 !important;
                width: 100% !important;
                font-size: 1.1rem !important;
            }
            div[data-testid="stButton"] > button[kind="secondary"]:hover {
                background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4) !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Create the chatbot button
            if st.button("💬 MediAssist AI Chatbot", key="chatbot_button", type="secondary"):
                # Open the chatbot in a new tab
                chatbot_url = "https://mediassist-ai-v6q9djfkdz73epitt6rrh9.streamlit.app/"
                # Create JavaScript to open in new tab
                js = f"""
                <script>
                    window.open("{chatbot_url}", "_blank");
                </script>
                """
                # Execute the JavaScript
                st.components.v1.html(js, height=0)
                # Show success message
                st.success("Opening MediAssist AI Chatbot...")
        
        st.markdown("""
        <div style="text-align: center; margin-top: 0.5rem;">
            <small style="color: #6B7280;">
                Get personalized medical advice
            </small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Medical disclaimer
        st.markdown("### ⚠️ Medical Disclaimer")
        st.caption("""
        This tool provides risk assessment only.
        Not a substitute for professional medical advice.
        Always consult healthcare providers for diagnosis.
        """)
    
    # Main content based on selection
    if app_mode == "🏠 Home":
        home_page()
    elif app_mode == "🩺 Liver Disease":
        liver_disease_prediction()
    elif app_mode == "🩺 Diabetes":
        diabetes_prediction()
    elif app_mode == "⚖️ Obesity":
        obesity_prediction()

if __name__ == "__main__":
    main()