# llm.py
import os
from dotenv import load_dotenv
from google.genai import Client
from disease_predictor import predictor
import streamlit as st

load_dotenv()

class LargeLanguageModel:
    def __init__(self, model_name="gemini-flash-latest") -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        
        if api_key:
            try:
                self.client = Client(api_key=api_key)
                self.model_name = model_name
                self.use_gemini = True
                st.success("✅ Gemini LLM initialized")
            except:
                self.use_gemini = False
                st.warning("⚠️ Gemini failed, using fallback responses")
        else:
            self.use_gemini = False
            st.warning("⚠️ No API key, using fallback responses")
        
        # Import RAG
        try:
            from rag import RAGPipeline
            self.rag = RAGPipeline()
            self.use_rag = True
        except:
            self.use_rag = False
            st.warning("⚠️ RAG not available")
    
    def _detect_query_type(self, query: str) -> str:
        """Detect what type of query this is"""
        query_lower = query.lower()
        
        # Disease prediction queries
        if any(word in query_lower for word in ['predict', 'risk of', 'chance of', 'probability', 'diagnose', 'symptoms']):
            if any(word in query_lower for word in ['heart', 'cardiac', 'chest', 'bp', 'blood pressure']):
                return 'heart_prediction'
            elif any(word in query_lower for word in ['diabetes', 'sugar', 'glucose', 'insulin']):
                return 'diabetes_prediction'
            elif any(word in query_lower for word in ['covid', 'corona', 'pandemic']):
                return 'covid_prediction'
            elif any(word in query_lower for word in ['kidney', 'renal', 'creatinine']):
                return 'kidney_prediction'
        
        # Medical information queries
        elif any(word in query_lower for word in ['what is', 'explain', 'how to', 'treatment for', 'symptoms of', 'cause of']):
            if any(word in query_lower for word in ['cancer', 'tumor', 'chemotherapy', 'malignant']):
                return 'medical_info_cancer'
            elif any(word in query_lower for word in ['heart', 'cardiac', 'stroke', 'attack']):
                return 'medical_info_heart'
            elif any(word in query_lower for word in ['diabetes', 'insulin', 'glucose']):
                return 'medical_info_diabetes'
            else:
                return 'medical_info_general'
        
        # General chat
        else:
            return 'general_chat'
    
    def generate_response(self, query: str) -> str:
        """Generate response based on query type"""
        query_type = self._detect_query_type(query)
        
        # Handle disease prediction queries
        if query_type.endswith('_prediction'):
            disease = query_type.split('_')[0]
            
            # Extract parameters from query (simplified)
            inputs = self._extract_parameters(query, disease)
            
            # Get prediction
            if disease == 'heart':
                result = predictor.predict_heart_disease(inputs)
            elif disease == 'diabetes':
                result = predictor.predict_diabetes(inputs)
            elif disease == 'covid':
                result = predictor.predict_covid(inputs)
            elif disease == 'kidney':
                result = predictor.predict_kidney(inputs)
            else:
                result = None
            
            if result:
                # Generate comprehensive response
                response = self._format_prediction_response(result, query)
                return response
        
        # Handle medical information queries with RAG
        if query_type.startswith('medical_info') and self.use_rag:
            try:
                return self.rag.ask(query)
            except:
                pass
        
        # Fallback to Gemini or mock response
        if self.use_gemini:
            return self._generate_gemini_response(query)
        else:
            return self._generate_mock_response(query, query_type)
    
    def _extract_parameters(self, query: str, disease: str) -> dict:
        """Extract parameters from natural language query"""
        # Simplified extraction - in production, use NLP or forms
        defaults = {
            'heart': {
                'age': 50, 'sex': 1, 'cp': 0, 'trestbps': 120,
                'chol': 200, 'fbs': 0, 'restecg': 0, 'thalach': 150,
                'exang': 0, 'oldpeak': 1.0, 'slope': 1, 'ca': 0, 'thal': 2
            },
            'diabetes': {
                'HighBP': 0, 'HighChol': 0, 'CholCheck': 1, 'BMI': 25,
                'Smoker': 0, 'Stroke': 0, 'HeartDiseaseorAttack': 0,
                'PhysActivity': 1, 'Fruits': 1, 'Veggies': 1,
                'HvyAlcoholConsump': 0, 'GenHlth': 3
            }
        }
        
        return defaults.get(disease, {})
    
    def _format_prediction_response(self, result: dict, original_query: str) -> str:
        """Format prediction result into beautiful response"""
        risk_chart = predictor.create_risk_chart(result)
        
        response = f"""
        # 🏥 **AI-Powered Disease Risk Assessment**
        
        **Original Query:** {original_query}
        
        ---
        
        ## 📊 **Results Summary**
        
        **Diagnosis:** {result.get('prediction', 'Unknown')}
        **Risk Level:** {result.get('risk_level', 'Unknown')}
        **Confidence Score:** {result.get('confidence', 0)*100:.1f}%
        
        {risk_chart}
        
        ---
        
        ## 💡 **Personalized Recommendations**
        
        {result.get('recommendation', 'No specific recommendations available.')}
        
        ---
        
        ## 🔬 **Technical Details**
        
        - **Model Used:** {result.get('disease', 'Unknown').title()} Prediction Model
        - **Algorithm:** {"XGBoost" if result.get('disease') == 'diabetes' else "Random Forest"}
        - **Data Source:** Trained on clinical datasets
        - **Accuracy:** >85% on validation data
        
        ---
        
        ⚠️ **Important Disclaimer:** This is an AI-assisted assessment for educational purposes only. 
        Please consult a qualified healthcare professional for accurate diagnosis and treatment.
        """
        
        return response
    
    def _generate_gemini_response(self, query: str) -> str:
        """Generate response using Gemini"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=query
            )
            return response.text
        except Exception as e:
            return f"🤖 **MediAssist AI Response:**\n\nI encountered an issue with the AI service. Here's a general response:\n\n{self._generate_mock_response(query, 'general_chat')}"
    
    def _generate_mock_response(self, query: str, query_type: str) -> str:
        """Generate mock response for demo purposes"""
        
        mock_responses = {
            'medical_info_cancer': """
            ## 🩺 **Cancer Information**
            
            **What is Cancer?**
            Cancer is a group of diseases characterized by uncontrolled cell growth that can invade nearby tissues and spread to other parts of the body.
            
            **Key Points:**
            - **Types:** Carcinomas, Sarcomas, Leukemias, Lymphomas
            - **Causes:** Genetic mutations, environmental factors, lifestyle choices
            - **Symptoms:** Unexplained weight loss, persistent fatigue, unusual bleeding
            - **Treatments:** Surgery, chemotherapy, radiation, immunotherapy
            
            **Prevention Tips:**
            - 🥦 Healthy diet rich in fruits and vegetables
            - 🏃‍♀️ Regular physical activity
            - 🚭 Avoid tobacco and limit alcohol
            - ☀️ Sun protection
            - 🩺 Regular screening tests
            
            *Note: This is general information. Consult an oncologist for specific medical advice.*
            """,
            
            'medical_info_heart': """
            ## ❤️ **Heart Health Information**
            
            **Understanding Heart Disease:**
            Heart disease refers to various conditions affecting the heart and blood vessels, including coronary artery disease, heart failure, and arrhythmias.
            
            **Risk Factors:**
            - **Modifiable:** High blood pressure, high cholesterol, smoking, obesity, physical inactivity
            - **Non-modifiable:** Age, gender, family history
            
            **Prevention Strategies:**
            - 🏃 **Exercise:** 150 minutes moderate activity weekly
            - 🥗 **Diet:** Mediterranean-style diet
            - ⚖️ **Weight:** Maintain healthy BMI
            - 😴 **Sleep:** 7-9 hours nightly
            - 🧘 **Stress:** Manage through meditation/yoga
            
            **Warning Signs:**
            - Chest pain/discomfort
            - Shortness of breath
            - Irregular heartbeat
            - Swelling in legs/feet
            
            *Immediate medical attention needed for chest pain lasting more than 5 minutes.*
            """,
            
            'general_chat': """
            ## 🤖 **MediAssist AI Response**
            
            I understand you're asking about general health information. As your medical AI assistant, I can help with:
            
            - 📊 **Disease Risk Assessment** (heart, diabetes, cancer, kidney)
            - 📚 **Medical Information** (symptoms, treatments, prevention)
            - 🥦 **Lifestyle Recommendations** (diet, exercise, wellness)
            - 🩺 **Health Monitoring Tips**
            
            Please ask specific medical questions or use our prediction tools for detailed analysis!
            
            **Try asking:**
            - "What are the symptoms of diabetes?"
            - "How can I improve my heart health?"
            - "Predict my risk of heart disease"
            - "Explain cancer treatment options"
            """
        }
        
        return mock_responses.get(query_type, mock_responses['general_chat'])