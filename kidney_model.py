import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class KidneyModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = 'ml_model/saved_models/kidney_model.pkl'
        self.scaler_path = 'ml_model/saved_models/kidney_scaler.pkl'
        
    def train(self, data_path='dataset/kidney.csv'):
        """Train the kidney disease prediction model"""
        # Load data
        df = pd.read_csv(data_path)
        
        # Prepare features and target
        X = df.drop('classification', axis=1)
        y = df['classification']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Save model and scaler
        os.makedirs('ml_model/saved_models', exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        return self.model
    
    def load_model(self):
        """Load trained model and scaler"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            return True
        return False
    
    def predict(self, input_data):
        """Make prediction on input data"""
        if self.model is None:
            if not self.load_model():
                self.train()
        
        # Scale input
        input_scaled = self.scaler.transform(input_data)
        
        # Make prediction
        prediction = self.model.predict(input_scaled)
        probability = self.model.predict_proba(input_scaled)
        
        return prediction[0], probability[0]
    
    def get_recommendations(self, prediction, probability, input_data):
        """Get personalized recommendations based on prediction"""
        # Handle probability array - it should have shape (2,) with [prob_class_0, prob_class_1]
        if len(probability) > 1:
            risk_score = probability[1] * 100  # Probability of class 1 (high risk)
        else:
            risk_score = probability[0] * 100 if prediction == 1 else (1 - probability[0]) * 100
        
        recommendations = {
            'risk_level': '',
            'risk_score': risk_score,
            'immediate_actions': [],
            'lifestyle_changes': [],
            'medical_advice': [],
            'prevention_tips': []
        }
        
        if prediction == 1:  # High risk of kidney disease
            recommendations['risk_level'] = 'High Risk'
            recommendations['immediate_actions'] = [
                '🚨 Consult a nephrologist (kidney specialist) immediately',
                '🩺 Get comprehensive kidney function tests (Creatinine, BUN, GFR)',
                '🔬 Complete urinalysis and urine protein test',
                '📊 Get ultrasound of kidneys and urinary tract',
                '💊 Review all current medications with your doctor',
                '📋 Check for diabetes and hypertension complications'
            ]
            recommendations['lifestyle_changes'] = [
                '💧 Increase water intake (8-10 glasses daily)',
                '🧂 Reduce sodium intake significantly (less than 2000mg/day)',
                '🥩 Limit protein intake as advised by doctor',
                '🚭 Quit smoking immediately',
                '🍷 Avoid alcohol consumption',
                '⚖️ Maintain healthy body weight',
                '🏃 Engage in light to moderate exercise',
                '😴 Get adequate rest (7-8 hours)'
            ]
            recommendations['medical_advice'] = [
                'Monitor blood pressure daily',
                'Regular kidney function monitoring (every 1-3 months)',
                'Discuss dialysis options if kidney function is very low',
                'Consider kidney transplant evaluation if needed',
                'Manage underlying conditions (diabetes, hypertension)',
                'Avoid NSAIDs and nephrotoxic medications',
                'Get vaccinated (flu, pneumonia vaccines)'
            ]
            recommendations['prevention_tips'] = [
                'Avoid foods high in potassium and phosphorus if advised',
                'Monitor fluid intake carefully',
                'Keep track of urine output',
                'Watch for swelling in legs, ankles, or face',
                'Be alert for symptoms: fatigue, nausea, confusion',
                'Maintain a kidney-friendly diet plan'
            ]
        else:  # Low risk
            recommendations['risk_level'] = 'Low Risk'
            recommendations['immediate_actions'] = [
                '✅ Your kidney function appears healthy',
                '📋 Continue regular health monitoring',
                '🎯 Maintain protective lifestyle habits',
                '💧 Keep up good hydration'
            ]
            recommendations['lifestyle_changes'] = [
                '💧 Drink adequate water (6-8 glasses daily)',
                '🥗 Eat balanced diet with fruits and vegetables',
                '🧂 Use salt moderately',
                '🏃 Maintain regular physical activity',
                '⚖️ Keep healthy body weight',
                '😴 Get quality sleep',
                '🧘 Manage stress effectively'
            ]
            recommendations['medical_advice'] = [
                'Annual kidney function screening after age 50',
                'Monitor if you have diabetes or hypertension',
                'Regular blood pressure checks',
                'Discuss family history with your doctor',
                'Get baseline kidney tests as recommended'
            ]
            recommendations['prevention_tips'] = [
                'Stay well-hydrated throughout the day',
                'Limit processed and high-sodium foods',
                'Avoid excessive use of pain medications',
                'Control blood sugar and blood pressure',
                'Avoid smoking and excessive alcohol',
                'Be aware of kidney disease symptoms'
            ]
        
        return recommendations

def get_feature_names():
    """Return feature names for kidney disease prediction"""
    return ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 
            'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 
            'cad', 'appet', 'pe', 'ane']