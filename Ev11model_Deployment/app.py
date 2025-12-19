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
    
    .sub-header {
        font-size: 2rem;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 700;
        border-bottom: 3px solid #3B82F6;
        padding-bottom: 0.5rem;
    }
    
    .model-card {
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        min-height: 300px;
    }
    
    .liver-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .diabetes-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .obesity-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        text-align: center;
        border-top: 4px solid #3B82F6;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #3B82F6, #1D4ED8);
        color: white;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
    }
    
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #D97706;
        margin: 1rem 0;
    }
    
    .feature-note {
        font-size: 0.85rem;
        color: #6B7280;
        font-style: italic;
    }
    
    .section-divider {
        height: 3px;
        background: linear-gradient(90deg, #3B82F6, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD PIPELINES ====================
@st.cache_resource
def load_liver_pipeline():
    """Load liver pipeline"""
    try:
        pipeline = joblib.load('Cleaned_Liver_disease.pkl')
        st.sidebar.success("✅ Liver model")
        return pipeline
    except:
        st.sidebar.error("❌ Liver model")
        return None

@st.cache_resource
def load_diabetes_pipeline():
    """Load diabetes pipeline"""
    try:
        pipeline = joblib.load('Cleaned_diabetes_disease.pkl')
        st.sidebar.success("✅ Diabetes model")
        return pipeline
    except:
        st.sidebar.error("❌ Diabetes model")
        return None

@st.cache_resource
def load_obesity_pipeline():
    """Load obesity pipeline"""
    try:
        pipeline = joblib.load('obesity_data_pipline.pkl')
        st.sidebar.success("✅ Obesity model")
        return pipeline
    except:
        st.sidebar.error("❌ Obesity model")
        return None

# ==================== LIVER FEATURE TRANSLATOR ====================
def translate_liver_features(user_inputs):
    """
    Translate user inputs to match model's expected features.
    
    Based on the error, your model expects:
    1. Age
    2. Total Bilirubin (was missing!)
    3. Direct Bilirubin
    4. Alkaline Phosphotase (with 'o', not 'a')
    5. Alamine Aminotransferase (ALT, not AST)
    6. Aspartate Aminotransferase (AST)
    7. Albumin
    8. Albumin and Globulin Ratio
    """
    
    # Default values for missing features
    translated = {
        'Age': user_inputs.get('age', 45),
        
        # Total Bilirubin is missing - estimate it (typically Direct is 20% of Total)
        'Total Bilirubin': user_inputs.get('total_bilirubin', 
                                         user_inputs.get('direct_bilirubin', 0.5) * 5),
        
        'Direct Bilirubin': user_inputs.get('direct_bilirubin', 0.5),
        
        # Note: Model expects "Alkaline Phosphotase" (with O)
        'Alkaline Phosphotase': user_inputs.get('alkaline_phosphatase', 187),
        
        # Alamine Aminotransferase (ALT) - different from AST
        # If user provided ALT, use it, otherwise estimate from AST
        'Alamine Aminotransferase': user_inputs.get('alamine_aminotransferase', 
                                                  user_inputs.get('aspartate_aminotransferase', 25)),
        
        'Aspartate Aminotransferase': user_inputs.get('aspartate_aminotransferase', 25),
        'Albumin': user_inputs.get('albumin', 4.5),
        'Albumin and Globulin Ratio': user_inputs.get('albumin_globulin_ratio', 1.2)
    }
    
    return translated

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
    
    # Warning about liver model features
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Important Note for Liver Model:</strong>
        Your liver model was trained with 8 features including:
        - <strong>Total Bilirubin</strong> (was missing before!)
        - <strong>Alamine Aminotransferase</strong> (ALT, different from AST)
        - <strong>Alkaline Phosphotase</strong> (with 'o', not 'a')
        
        The app now automatically handles these differences.
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
    
    # Information about the model
    with st.expander("ℹ️ Important: Your Model Expects These 8 Features", expanded=True):
        st.markdown("""
        **Your liver model was trained with these EXACT 8 features:**
        1. `Age`
        2. `Total Bilirubin` *(was missing before!)*
        3. `Direct Bilirubin`
        4. `Alkaline Phosphotase` *(with 'o', not 'a')*
        5. `Alamine Aminotransferase` *(ALT, different from AST)*
        6. `Aspartate Aminotransferase` *(AST)*
        7. `Albumin`
        8. `Albumin and Globulin Ratio`
        
        **Key Differences:**
        - **ALT vs AST:** ALT is more liver-specific, AST is found in liver/heart/muscles
        - **Total vs Direct Bilirubin:** Total = Direct + Indirect
        - **Phosphotase vs Phosphatase:** Spelling difference in model
        
        **The app automatically translates your inputs to match the model.**
        """)
    
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
        st.markdown('<p class="feature-note">Normal: 0.2-1.2 mg/dL</p>', unsafe_allow_html=True)
        
        direct_bilirubin = st.number_input("Direct Bilirubin (mg/dL)",
                                          min_value=0.0, max_value=20.0, value=0.5, step=0.1,
                                          help="Normal: 0.1-0.3 mg/dL")
        st.markdown('<p class="feature-note">Normal: 0.1-0.3 mg/dL</p>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Liver Enzymes")
        
        st.markdown("**Alkaline Phosphotase** *(note spelling with 'o')*")
        alkaline_phosphatase = st.number_input("Alkaline Phosphatase (IU/L)", 
                                              min_value=20, max_value=1400, value=187, step=1,
                                              help="Model expects 'Phosphotase'", key="alp_input")
        st.markdown('<p class="feature-note">Model expects: Alkaline Phosphotase</p>', unsafe_allow_html=True)
        
        st.markdown("**Alamine Aminotransferase (ALT)**")
        alamine_aminotransferase = st.number_input("Alamine Aminotransferase (ALT) (IU/L)", 
                                                  min_value=5, max_value=500, value=30, step=1,
                                                  help="Liver-specific enzyme, Normal: 7-56 IU/L")
        st.markdown('<p class="feature-note">Normal: 7-56 IU/L (ALT)</p>', unsafe_allow_html=True)
        
        st.markdown("**Aspartate Aminotransferase (AST)**")
        aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase (AST) (IU/L)",
                                                    min_value=5, max_value=500, value=25, step=1,
                                                    help="Found in liver/heart/muscles, Normal: 10-40 IU/L")
        st.markdown('<p class="feature-note">Normal: 10-40 IU/L (AST)</p>', unsafe_allow_html=True)
    
    with col3:
        st.markdown("#### Protein Levels")
        albumin = st.number_input("Albumin (g/dL)", min_value=1.0, max_value=6.0, value=4.5, step=0.1)
        st.markdown('<p class="feature-note">Normal: 3.5-5.5 g/dL</p>', unsafe_allow_html=True)
        
        albumin_globulin_ratio = st.number_input("Albumin/Globulin Ratio", 
                                                min_value=0.1, max_value=3.0, value=1.2, step=0.1)
        st.markdown('<p class="feature-note">Normal: 1.1-2.5</p>', unsafe_allow_html=True)
        
        # Show calculated indirect bilirubin
        indirect_bilirubin = total_bilirubin - direct_bilirubin
        st.info(f"**Indirect Bilirubin (calculated):** {indirect_bilirubin:.2f} mg/dL")
        
        # Show AST/ALT ratio if available
        if alamine_aminotransferase > 0:
            ast_alt_ratio = aspartate_aminotransferase / alamine_aminotransferase
            st.info(f"**AST/ALT Ratio:** {ast_alt_ratio:.2f}")
    
    # Show what features will be sent to model
    with st.expander("🔍 View Feature Translation", expanded=False):
        user_inputs = {
            'age': age,
            'total_bilirubin': total_bilirubin,
            'direct_bilirubin': direct_bilirubin,
            'alkaline_phosphatase': alkaline_phosphatase,
            'alamine_aminotransferase': alamine_aminotransferase,
            'aspartate_aminotransferase': aspartate_aminotransferase,
            'albumin': albumin,
            'albumin_globulin_ratio': albumin_globulin_ratio
        }
        
        translated = translate_liver_features(user_inputs)
        
        st.write("**Your Inputs:**")
        st.json(user_inputs)
        
        st.write("**Translated for Model:**")
        st.json(translated)
    
    # Prediction button
    if st.button("📊 Predict Liver Disease Risk", type="primary"):
        with st.spinner("Translating features and analyzing..."):
            try:
                # Collect user inputs
                user_inputs = {
                    'age': age,
                    'total_bilirubin': total_bilirubin,
                    'direct_bilirubin': direct_bilirubin,
                    'alkaline_phosphatase': alkaline_phosphatase,
                    'alamine_aminotransferase': alamine_aminotransferase,
                    'aspartate_aminotransferase': aspartate_aminotransferase,
                    'albumin': albumin,
                    'albumin_globulin_ratio': albumin_globulin_ratio
                }
                
                # Translate to model's expected features
                model_features = translate_liver_features(user_inputs)
                
                # Try different naming variations
                feature_variations = [
                    model_features,  # Try exact translation first
                    
                    # Try with underscores
                    {
                        'Age': age,
                        'Total_Bilirubin': total_bilirubin,
                        'Direct_Bilirubin': direct_bilirubin,
                        'Alkaline_Phosphotase': alkaline_phosphatase,
                        'Alamine_Aminotransferase': alamine_aminotransferase,
                        'Aspartate_Aminotransferase': aspartate_aminotransferase,
                        'Albumin': albumin,
                        'Albumin_and_Globulin_Ratio': albumin_globulin_ratio
                    },
                    
                    # Try with lowercase
                    {
                        'age': age,
                        'total bilirubin': total_bilirubin,
                        'direct bilirubin': direct_bilirubin,
                        'alkaline phosphotase': alkaline_phosphatase,
                        'alamine aminotransferase': alamine_aminotransferase,
                        'aspartate aminotransferase': aspartate_aminotransferase,
                        'albumin': albumin,
                        'albumin and globulin ratio': albumin_globulin_ratio
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
                    except Exception as e:
                        continue
                
                if not prediction_made:
                    st.error("Feature matching failed. Please check model features.")
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
                
                # Clinical interpretation
                st.markdown("### 🔍 Clinical Interpretation")
                
                abnormal_count = 0
                if total_bilirubin > 1.2: abnormal_count += 1
                if direct_bilirubin > 0.3: abnormal_count += 1
                if alkaline_phosphatase > 147: abnormal_count += 1
                if alamine_aminotransferase > 56: abnormal_count += 1
                if aspartate_aminotransferase > 40: abnormal_count += 1
                if albumin < 3.5: abnormal_count += 1
                
                st.write(f"**{abnormal_count} out of 7 biomarkers are abnormal**")
                
                # Store history
                if 'liver_history' not in st.session_state:
                    st.session_state.liver_history = []
                
                st.session_state.liver_history.append({
                    'timestamp': datetime.now().strftime("%H:%M"),
                    'diagnosis': diagnosis,
                    'probability': f"{disease_prob:.1f}%",
                    'abnormal_tests': abnormal_count
                })
                
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
                
                # Store history
                if 'diabetes_history' not in st.session_state:
                    st.session_state.diabetes_history = []
                
                st.session_state.diabetes_history.append({
                    'timestamp': datetime.now().strftime("%H:%M"),
                    'diagnosis': diagnosis,
                    'probability': f"{diabetes_prob:.1f}%"
                })
                
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
                
                # Try to get probabilities
                try:
                    probabilities = pipeline.predict_proba(input_df)[0]
                except:
                    probabilities = None
                
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
                    if probabilities is not None:
                        top_prob = max(probabilities) * 100
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Confidence", f"{top_prob:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Store history
                if 'obesity_history' not in st.session_state:
                    st.session_state.obesity_history = []
                
                st.session_state.obesity_history.append({
                    'timestamp': datetime.now().strftime("%H:%M"),
                    'category': predicted_category,
                    'bmi': f"{bmi:.1f}",
                    'age': age
                })
                
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
        
        # Load models for status
        st.markdown("### 📊 Model Status")
        liver_model = load_liver_pipeline()
        diabetes_model = load_diabetes_pipeline()
        obesity_model = load_obesity_pipeline()
        
        if liver_model:
            st.success("✅ Liver (8 features)")
        if diabetes_model:
            st.success("✅ Diabetes (11 features)")
        if obesity_model:
            st.success("✅ Obesity (6 features)")
        
        st.markdown("---")
        st.markdown("### ⚠️ Medical Disclaimer")
        st.caption("Risk assessment only. Consult healthcare providers.")
    
    # Main content
    if app_mode == "🏠 Home":
        home_page()
    elif app_mode == "🩺 Liver Disease":
        liver_disease_prediction()
    elif app_mode == "🩺 Diabetes":
        diabetes_prediction()
    elif app_mode == "⚖️ Obesity":
        obesity_prediction()

if __name__ == "__main__":
    # Initialize session state
    if 'liver_history' not in st.session_state:
        st.session_state.liver_history = []
    if 'diabetes_history' not in st.session_state:
        st.session_state.diabetes_history = []
    if 'obesity_history' not in st.session_state:
        st.session_state.obesity_history = []
    
    main()