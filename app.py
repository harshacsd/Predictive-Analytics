import streamlit as st
import pandas as pd
import numpy as np
from ml_model.diabetes_model import DiabetesModel, get_feature_names as get_diabetes_features
from ml_model.heart_model import HeartModel, get_feature_names as get_heart_features
from ml_model.kidney_model import KidneyModel, get_feature_names as get_kidney_features
from ml_model.stroke_model import StrokeModel, get_feature_names as get_stroke_features
from ml_model.hypertension_model import HypertensionModel, get_feature_names as get_hypertension_features
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Healthcare Predictive Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Bootstrap CSS and Custom Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .stApp {
        background: transparent;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
        animation: slideDown 0.5s ease-out;
    }
    
    @keyframes slideDown {
        from {
            transform: translateY(-50px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Card Styling */
    .custom-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: none;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    .card-header-custom {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
    }
    
    /* Feature Cards */
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s ease;
        border: 2px solid transparent;
        height: 100%;
    }
    
    .feature-card:hover {
        border-color: #667eea;
        transform: scale(1.05);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    
    .feature-card i {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .feature-card h3 {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Input Fields */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 1rem 3rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    /* Alert Boxes */
    .alert-custom {
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(-50px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .alert-success {
        background: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    
    .alert-danger {
        background: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
    
    .alert-info {
        background: #d1ecf1;
        border-color: #17a2b8;
        color: #0c5460;
    }
    
    .alert-warning {
        background: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }
    
    /* Recommendations Section */
    .recommendation-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .recommendation-box h3 {
        color: #667eea;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .recommendation-box ul {
        list-style: none;
        padding: 0;
    }
    
    .recommendation-box li {
        padding: 0.5rem 0;
        border-bottom: 1px solid #f0f0f0;
        position: relative;
        padding-left: 1.5rem;
    }
    
    .recommendation-box li:before {
        content: "→";
        position: absolute;
        left: 0;
        color: #667eea;
        font-weight: bold;
    }
    
    .recommendation-box li:last-child {
        border-bottom: none;
    }
    
    /* Footer */
    .footer {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 3rem;
        text-align: center;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem;
        }
        
        .custom-card {
            padding: 1rem;
        }
        
        .feature-card {
            padding: 1rem;
        }
    }
    </style>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <div class="container">
            <h1><i class="fas fa-hospital-alt"></i> Healthcare Predictive Analytics System</h1>
            <p>Disease Risk Assessment & Personalized Health Recommendations</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <i class="fas fa-heartbeat" style="font-size: 4rem; color: white;"></i>
            <h2 style="color: white; margin-top: 1rem;">Navigation</h2>
        </div>
    """, unsafe_allow_html=True)
    
    disease_option = st.selectbox(
        "Select Disease Prediction",
        ["Home", "Diabetes", "Heart Disease", "Kidney Disease", "Stroke", "Hypertension"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
        <div style="color: white; padding: 1rem;">
            <h3><i class="fas fa-info-circle"></i> About</h3>
            <p>This application uses advanced machine learning algorithms to predict disease risks and provides personalized recommendations.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; color: white;">
            <h3><i class="fas fa-exclamation-triangle"></i> Disclaimer</h3>
            <p style="font-size: 0.9rem;">This tool is for informational purposes only. Always consult with healthcare professionals.</p>
        </div>
    """, unsafe_allow_html=True)

# Home Page
if disease_option == "Home":
    st.markdown("""
        <div class="custom-card">
            <div class="card-header-custom">
                <i class="fas fa-home"></i> Welcome to Healthcare Predictive Analytics
            </div>
            <p style="font-size: 1.1rem; color: #555; line-height: 1.8;">
                Analyzes your health data to predict disease risks and provide personalized recommendations.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <i class="fas fa-user-md"></i>
                <h3>What We Offer</h3>
                <ul style="text-align: left; list-style: none; padding: 0;">
                    <li><i class="fas fa-check" style="color: #667eea;"></i> Disease Risk Prediction</li>
                    <li><i class="fas fa-check" style="color: #667eea;"></i> Personalized Recommendations</li>
                    <li><i class="fas fa-check" style="color: #667eea;"></i> Prevention Strategies</li>
                    <li><i class="fas fa-check" style="color: #667eea;"></i> Medical Guidance</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <i class="fas fa-disease"></i>
                <h3>Available Predictions</h3>
                <ul style="text-align: left; list-style: none; padding: 0;">
                    <li><i class="fas fa-syringe" style="color: #667eea;"></i> Diabetes Risk</li>
                    <li><i class="fas fa-heart" style="color: #667eea;"></i> Heart Disease Risk</li>
                    <li><i class="fas fa-kidneys" style="color: #667eea;"></i> Kidney Disease Risk</li>
                    <li><i class="fas fa-brain" style="color: #667eea;"></i> Stroke Risk</li>
                    <li><i class="fas fa-heartbeat" style="color: #667eea;"></i> Hypertension Risk</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="feature-card">
                <i class="fas fa-cogs"></i>
                <h3>How It Works</h3>
                <ul style="text-align: left; list-style: none; padding: 0;">
                    <li><i class="fas fa-arrow-right" style="color: #667eea;"></i> Select a disease type</li>
                    <li><i class="fas fa-arrow-right" style="color: #667eea;"></i> Enter your health data</li>
                    <li><i class="fas fa-arrow-right" style="color: #667eea;"></i> Get instant predictions</li>
                    <li><i class="fas fa-arrow-right" style="color: #667eea;"></i> Receive recommendations</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="alert-custom alert-info">
            <h4><i class="fas fa-arrow-left"></i> Get Started</h4>
            <p>Select a disease prediction from the sidebar to begin your health assessment.</p>
        </div>
    """, unsafe_allow_html=True)

# Diabetes Prediction
elif disease_option == "Diabetes":
    st.markdown("""
        <div class="custom-card">
            <div class="card-header-custom">
                <i class="fas fa-syringe"></i> Diabetes Risk Prediction
            </div>
            <p style="color: #666;">Enter your health information to assess your diabetes risk.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=80)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    
    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
    
    if st.button("🔍 Predict Diabetes Risk", type="primary"):
        with st.spinner("Analyzing your data..."):
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                   insulin, bmi, dpf, age]])
            
            model = DiabetesModel()
            prediction, probability = model.predict(input_data)
            recommendations = model.get_recommendations(prediction, probability, input_data)
            
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("<h2 style='color: #667eea;'><i class='fas fa-chart-line'></i> Prediction Results</h2>", unsafe_allow_html=True)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=recommendations['risk_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Score", 'font': {'size': 24, 'color': '#667eea'}},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if prediction == 1 else "green"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            if prediction == 1:
                st.markdown(f"""
                    <div class="alert-custom alert-danger">
                        <h3><i class="fas fa-exclamation-triangle"></i> {recommendations['risk_level']}</h3>
                        <p>Diabetes Risk Detected</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="alert-custom alert-success">
                        <h3><i class="fas fa-check-circle"></i> {recommendations['risk_level']}</h3>
                        <p>Low Diabetes Risk</p>
                    </div>
                """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='recommendation-box'><h3><i class='fas fa-exclamation-circle'></i> Immediate Actions</h3><ul>", unsafe_allow_html=True)
                for action in recommendations['immediate_actions']:
                    st.markdown(f"<li>{action}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
                
                st.markdown("<div class='recommendation-box'><h3><i class='fas fa-utensils'></i> Lifestyle Changes</h3><ul>", unsafe_allow_html=True)
                for change in recommendations['lifestyle_changes']:
                    st.markdown(f"<li>{change}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='recommendation-box'><h3><i class='fas fa-pills'></i> Medical Advice</h3><ul>", unsafe_allow_html=True)
                for advice in recommendations['medical_advice']:
                    st.markdown(f"<li>{advice}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
                
                st.markdown("<div class='recommendation-box'><h3><i class='fas fa-shield-alt'></i> Prevention Tips</h3><ul>", unsafe_allow_html=True)
                for tip in recommendations['prevention_tips']:
                    st.markdown(f"<li>{tip}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

# Heart Disease Prediction
elif disease_option == "Heart Disease":
    st.markdown("""
        <div class="custom-card">
            <div class="card-header-custom">
                <i class="fas fa-heart"></i> Heart Disease Risk Prediction
            </div>
            <p style="color: #666;">Enter your cardiovascular health information.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        sex_val = 1 if sex == "Male" else 0
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        cp_val = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=250, value=120)
        chol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=600, value=200)
    
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
        fbs_val = 1 if fbs == "Yes" else 0
        restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "LV Hypertrophy"])
        restecg_val = ["Normal", "ST-T Abnormality", "LV Hypertrophy"].index(restecg)
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250, value=150)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        exang_val = 1 if exang == "Yes" else 0
    
    with col3:
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST", ["Upsloping", "Flat", "Downsloping"])
        slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope)
        ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
        thal_val = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 1
    
    if st.button("🔍 Predict Heart Disease Risk", type="primary"):
        with st.spinner("Analyzing your cardiovascular health..."):
            input_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val, 
                                   thalach, exang_val, oldpeak, slope_val, ca, thal_val]])
            
            model = HeartModel()
            prediction, probability = model.predict(input_data)
            recommendations = model.get_recommendations(prediction, probability, input_data)
            
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("<h2 style='color: #667eea;'><i class='fas fa-chart-line'></i> Prediction Results</h2>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=recommendations['risk_score'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Score", 'font': {'size': 20}},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if prediction == 1 else "green"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ]
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric("Max Heart Rate", f"{recommendations['heart_analysis']['max_heart_rate']} bpm")
                st.metric("Expected Max HR", f"{recommendations['heart_analysis']['expected_max']} bpm")
            
            with col3:
                st.metric("Heart Rate %", f"{recommendations['heart_analysis']['percentage']:.1f}%")
                st.info(f"**Status:** {recommendations['heart_analysis']['status']}")
            
            if prediction == 1:
                st.markdown(f"""
                    <div class="alert-custom alert-danger">
                        <h3><i class="fas fa-exclamation-triangle"></i> {recommendations['risk_level']}</h3>
                        <p>Heart Disease Risk Detected</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="alert-custom alert-success">
                        <h3><i class="fas fa-check-circle"></i> {recommendations['risk_level']}</h3>
                        <p>Low Heart Disease Risk</p>
                    </div>
                """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='recommendation-box'><h3><i class='fas fa-exclamation-circle'></i> Immediate Actions</h3><ul>", unsafe_allow_html=True)
                for action in recommendations['immediate_actions']:
                    st.markdown(f"<li>{action}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
                
                st.markdown("<div class='recommendation-box'><h3><i class='fas fa-utensils'></i> Lifestyle Changes</h3><ul>", unsafe_allow_html=True)
                for change in recommendations['lifestyle_changes']:
                    st.markdown(f"<li>{change}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='recommendation-box'><h3><i class='fas fa-pills'></i> Medical Advice</h3><ul>", unsafe_allow_html=True)
                for advice in recommendations['medical_advice']:
                    st.markdown(f"<li>{advice}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
                
                st.markdown("<div class='recommendation-box'><h3><i class='fas fa-shield-alt'></i> Prevention Tips</h3><ul>", unsafe_allow_html=True)
                for tip in recommendations['prevention_tips']:
                    st.markdown(f"<li>{tip}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

# Kidney Disease Prediction
elif disease_option == "Kidney Disease":
    st.markdown("""
        <div class="custom-card">
            <div class="card-header-custom">
                <i class="fas fa-kidneys"></i> Kidney Disease Risk Prediction
            </div>
            <p style="color: #666;">Enter your kidney function parameters.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        bp = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=80)
        sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
        al = st.number_input("Albumin (0-5)", min_value=0, max_value=5, value=0)
        su = st.number_input("Sugar (0-5)", min_value=0, max_value=5, value=0)
        rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
        rbc_val = 1 if rbc == "Normal" else 0
        pc = st.selectbox("Pus Cell", ["Normal", "Abnormal"])
        pc_val = 1 if pc == "Normal" else 0
        pcc = st.selectbox("Pus Cell Clumps", ["Not Present", "Present"])
        pcc_val = 1 if pcc == "Present" else 0
    
    with col2:
        ba = st.selectbox("Bacteria", ["Not Present", "Present"])
        ba_val = 1 if ba == "Present" else 0
        bgr = st.number_input("Blood Glucose Random (mg/dL)", min_value=0, max_value=500, value=120)
        bu = st.number_input("Blood Urea (mg/dL)", min_value=0, max_value=200, value=40)
        sc = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, max_value=15.0, value=1.0, step=0.1)
        sod = st.number_input("Sodium (mEq/L)", min_value=0, max_value=200, value=135)
        pot = st.number_input("Potassium (mEq/L)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
        hemo = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
        pcv = st.number_input("Packed Cell Volume (%)", min_value=0, max_value=60, value=40)
    
    with col3:
        wc = st.number_input("White Blood Cell Count", min_value=0, max_value=30000, value=7000)
        rc = st.number_input("Red Blood Cell Count (millions/cmm)", min_value=0.0, max_value=10.0, value=4.5, step=0.1)
        htn = st.selectbox("Hypertension", ["No", "Yes"])
        htn_val = 1 if htn == "Yes" else 0
        dm = st.selectbox("Diabetes Mellitus", ["No", "Yes"])
        dm_val = 1 if dm == "Yes" else 0
        cad = st.selectbox("Coronary Artery Disease", ["No", "Yes"])
        cad_val = 1 if cad == "Yes" else 0
        appet = st.selectbox("Appetite", ["Good", "Poor"])
        appet_val = 1 if appet == "Good" else 0
        pe = st.selectbox("Pedal Edema", ["No", "Yes"])
        pe_val = 1 if pe == "Yes" else 0
        ane = st.selectbox("Anemia", ["No", "Yes"])
        ane_val = 1 if ane == "Yes" else 0
    
    if st.button("🔍 Predict Kidney Disease Risk", type="primary"):
        with st.spinner("Analyzing your kidney function..."):
            input_data = np.array([[age, bp, sg, al, su, rbc_val, pc_val, pcc_val, ba_val, bgr, 
                                   bu, sc, sod, pot, hemo, pcv, wc, rc, htn_val, dm_val, cad_val, 
                                   appet_val, pe_val, ane_val]])
            
            model = KidneyModel()
            prediction, probability = model.predict(input_data)
            recommendations = model.get_recommendations(prediction, probability, input_data)
            
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("<h2 style='color: #667eea;'><i class='fas fa-chart-line'></i> Prediction Results</h2>", unsafe_allow_html=True)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=recommendations['risk_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Score", 'font': {'size': 24, 'color': '#667eea'}},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if prediction == 1 else "green"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            if prediction == 1:
                st.markdown(f"""
                    <div class="alert-custom alert-danger">
                        <h3><i class="fas fa-exclamation-triangle"></i> {recommendations['risk_level']}</h3>
                        <p>Kidney Disease Risk Detected</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="alert-custom alert-success">
                        <h3><i class="fas fa-check-circle"></i> {recommendations['risk_level']}</h3>
                        <p>Low Kidney Disease Risk</p>
                    </div>
                """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='recommendation-box'><h3><i class='fas fa-exclamation-circle'></i> Immediate Actions</h3><ul>", unsafe_allow_html=True)
                for action in recommendations['immediate_actions']:
                    st.markdown(f"<li>{action}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
                
                st.markdown("<div class='recommendation-box'><h3><i class='fas fa-utensils'></i> Lifestyle Changes</h3><ul>", unsafe_allow_html=True)
                for change in recommendations['lifestyle_changes']:
                    st.markdown(f"<li>{change}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='recommendation-box'><h3><i class='fas fa-pills'></i> Medical Advice</h3><ul>", unsafe_allow_html=True)
                for advice in recommendations['medical_advice']:
                    st.markdown(f"<li>{advice}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
                
                st.markdown("<div class='recommendation-box'><h3><i class='fas fa-shield-alt'></i> Prevention Tips</h3><ul>", unsafe_allow_html=True)
                for tip in recommendations['prevention_tips']:
                    st.markdown(f"<li>{tip}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

# Stroke Prediction
elif disease_option == "Stroke":
    st.markdown("""
        <div class="custom-card">
            <div class="card-header-custom">
                <i class="fas fa-brain"></i> Stroke Risk Prediction
            </div>
            <p style="color: #666;">Enter your health information to assess stroke risk.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        hypertension_val = 1 if hypertension == "Yes" else 0
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        heart_disease_val = 1 if heart_disease == "Yes" else 0
        ever_married = st.selectbox("Ever Married", ["No", "Yes"])
    
    with col2:
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        avg_glucose = st.number_input("Average Glucose Level (mg/dL)", min_value=0.0, max_value=300.0, value=100.0, step=0.1)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
    
    if st.button("🔍 Predict Stroke Risk", type="primary"):
        with st.spinner("Analyzing your stroke risk..."):
            temp_model = StrokeModel()
            temp_model.train()
            
            gender_encoded = temp_model.label_encoders['gender'].transform([gender])[0]
            ever_married_encoded = temp_model.label_encoders['ever_married'].transform([ever_married])[0]
            work_type_encoded = temp_model.label_encoders['work_type'].transform([work_type])[0]
            residence_encoded = temp_model.label_encoders['Residence_type'].transform([residence_type])[0]
            smoking_encoded = temp_model.label_encoders['smoking_status'].transform([smoking_status])[0]
            
            input_data = np.array([[gender_encoded, age, hypertension_val, heart_disease_val, 
                                   ever_married_encoded, work_type_encoded, residence_encoded, 
                                   avg_glucose, bmi, smoking_encoded]])
            
            prediction, probability = temp_model.predict(input_data)
            recommendations = temp_model.get_recommendations(prediction, probability, input_data)
            
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("<h2 style='color: #667eea;'><i class='fas fa-chart-line'></i> Prediction Results</h2>", unsafe_allow_html=True)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=recommendations['risk_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Score", 'font': {'size': 24, 'color': '#667eea'}},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if prediction == 1 else "green"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            if prediction == 1:
                st.markdown(f"""
                    <div class="alert-custom alert-danger">
                        <h3><i class="fas fa-exclamation-triangle"></i> {recommendations['risk_level']}</h3>
                        <p>Stroke Risk Detected</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="alert-custom alert-success">
                        <h3><i class="fas fa-check-circle"></i> {recommendations['risk_level']}</h3>
                        <p>Low Stroke Risk</p>
                    </div>
                """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='recommendation-box'><h3><i class='fas fa-exclamation-circle'></i> Immediate Actions</h3><ul>", unsafe_allow_html=True)
                for action in recommendations['immediate_actions']:
                    st.markdown(f"<li>{action}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
                
                st.markdown("<div class='recommendation-box'><h3><i class='fas fa-utensils'></i> Lifestyle Changes</h3><ul>", unsafe_allow_html=True)
                for change in recommendations['lifestyle_changes']:
                    st.markdown(f"<li>{change}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='recommendation-box'><h3><i class='fas fa-pills'></i> Medical Advice</h3><ul>", unsafe_allow_html=True)
                for advice in recommendations['medical_advice']:
                    st.markdown(f"<li>{advice}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
                
                st.markdown("<div class='recommendation-box'><h3><i class='fas fa-shield-alt'></i> Prevention Tips</h3><ul>", unsafe_allow_html=True)
                for tip in recommendations['prevention_tips']:
                    st.markdown(f"<li>{tip}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

# Hypertension Prediction
elif disease_option == "Hypertension":
    st.markdown("""
        <div class="custom-card">
            <div class="card-header-custom">
                <i class="fas fa-heartbeat"></i> Hypertension Risk Prediction
            </div>
            <p style="color: #666;">Enter your cardiovascular parameters.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        sex_val = 1 if sex == "Male" else 0
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        cp_val = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=250, value=120)
        chol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=600, value=200)
    
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
        fbs_val = 1 if fbs == "Yes" else 0
        restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "LV Hypertrophy"])
        restecg_val = ["Normal", "ST-T Abnormality", "LV Hypertrophy"].index(restecg)
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250, value=150)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        exang_val = 1 if exang == "Yes" else 0
    
    with col3:
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST", ["Upsloping", "Flat", "Downsloping"])
        slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope)
        ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
        thal_val = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 1
    
    if st.button("🔍 Predict Hypertension Risk", type="primary"):
        with st.spinner("Analyzing your blood pressure risk..."):
            input_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val, 
                                   thalach, exang_val, oldpeak, slope_val, ca, thal_val]])
            
            model = HypertensionModel()
            prediction, probability = model.predict(input_data)
            recommendations = model.get_recommendations(prediction, probability, input_data)
            
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("<h2 style='color: #667eea;'><i class='fas fa-chart-line'></i> Prediction Results</h2>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=recommendations['risk_score'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Score", 'font': {'size': 20}},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if prediction == 1 else "green"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ]
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric("Systolic BP", f"{recommendations['bp_analysis']['systolic_bp']} mm Hg")
                st.info(f"**Status:** {recommendations['bp_analysis']['status']}")
            
            with col3:
                st.markdown("### 🩺 BP Category")
                st.info(recommendations['bp_analysis']['category'])
            
            if prediction == 1:
                st.markdown(f"""
                    <div class="alert-custom alert-danger">
                        <h3><i class="fas fa-exclamation-triangle"></i> {recommendations['risk_level']}</h3>
                        <p>Hypertension Risk Detected</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="alert-custom alert-success">
                        <h3><i class="fas fa-check-circle"></i> {recommendations['risk_level']}</h3>
                        <p>Low Hypertension Risk</p>
                    </div>
                """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='recommendation-box'><h3><i class='fas fa-exclamation-circle'></i> Immediate Actions</h3><ul>", unsafe_allow_html=True)
                for action in recommendations['immediate_actions']:
                    st.markdown(f"<li>{action}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
                
                st.markdown("<div class='recommendation-box'><h3><i class='fas fa-utensils'></i> Lifestyle Changes</h3><ul>", unsafe_allow_html=True)
                for change in recommendations['lifestyle_changes']:
                    st.markdown(f"<li>{change}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='recommendation-box'><h3><i class='fas fa-pills'></i> Medical Advice</h3><ul>", unsafe_allow_html=True)
                for advice in recommendations['medical_advice']:
                    st.markdown(f"<li>{advice}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
                
                st.markdown("<div class='recommendation-box'><h3><i class='fas fa-shield-alt'></i> Prevention Tips</h3><ul>", unsafe_allow_html=True)
                for tip in recommendations['prevention_tips']:
                    st.markdown(f"<li>{tip}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <h3><i class="fas fa-hospital-alt"></i> Healthcare Predictive Analytics System</h3>
        <p>Powered by Advanced Machine Learning & AI</p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">
            <i class="fas fa-exclamation-triangle"></i> 
            This is a predictive tool. Always consult healthcare professionals for accurate diagnosis and treatment.
        </p>
        <div style="margin-top: 1.5rem;">
            <i class="fab fa-facebook" style="font-size: 1.5rem; margin: 0 0.5rem; cursor: pointer;"></i>
            <i class="fab fa-twitter" style="font-size: 1.5rem; margin: 0 0.5rem; cursor: pointer;"></i>
            <i class="fab fa-linkedin" style="font-size: 1.5rem; margin: 0 0.5rem; cursor: pointer;"></i>
            <i class="fab fa-instagram" style="font-size: 1.5rem; margin: 0 0.5rem; cursor: pointer;"></i>
        </div>
    </div>
""", unsafe_allow_html=True)