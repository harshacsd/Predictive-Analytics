import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class HypertensionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = 'ml_model/saved_models/hypertension_model.pkl'
        self.scaler_path = 'ml_model/saved_models/hypertension_scaler.pkl'
        
    def train(self, data_path='dataset/hypertension.csv'):
        """Train the hypertension prediction model"""
        # Load data
        df = pd.read_csv(data_path)
        
        # Prepare features and target
        X = df.drop('hypertension', axis=1)
        y = df['hypertension']
        
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
        
        # Extract blood pressure
        trestbps = input_data[0][3]  # Resting blood pressure
        
        recommendations = {
            'risk_level': '',
            'risk_score': risk_score,
            'bp_analysis': {},
            'immediate_actions': [],
            'lifestyle_changes': [],
            'medical_advice': [],
            'prevention_tips': []
        }
        
        # Blood pressure classification
        if trestbps < 120:
            bp_status = 'Normal'
            bp_category = 'Optimal blood pressure'
        elif trestbps < 130:
            bp_status = 'Elevated'
            bp_category = 'Pre-hypertension - needs attention'
        elif trestbps < 140:
            bp_status = 'Stage 1 Hypertension'
            bp_category = 'High blood pressure - medical intervention needed'
        else:
            bp_status = 'Stage 2 Hypertension'
            bp_category = 'Very high blood pressure - urgent medical attention'
        
        recommendations['bp_analysis'] = {
            'systolic_bp': trestbps,
            'status': bp_status,
            'category': bp_category
        }
        
        if prediction == 1:  # High risk of hypertension
            recommendations['risk_level'] = 'High Risk'
            
            if trestbps >= 140:
                recommendations['immediate_actions'] = [
                    '🚨 URGENT: Consult your doctor immediately - you have Stage 2 Hypertension',
                    '🏥 Get comprehensive cardiovascular evaluation',
                    '📊 24-hour ambulatory blood pressure monitoring',
                    '💊 Discuss immediate medication options',
                    '🔬 Complete blood work (kidney function, electrolytes)',
                    '🩺 ECG and echocardiogram to check heart health'
                ]
            else:
                recommendations['immediate_actions'] = [
                    '🚨 Schedule appointment with your doctor urgently',
                    '📊 Start monitoring blood pressure twice daily',
                    '🩺 Get complete cardiovascular health assessment',
                    '🔬 Blood tests for kidney function and cholesterol',
                    '💊 Discuss preventive medication with doctor'
                ]
            
            recommendations['lifestyle_changes'] = [
                '🧂 Reduce sodium intake to less than 1500mg per day',
                '🥗 Adopt DASH diet (Dietary Approaches to Stop Hypertension)',
                '⚖️ Lose weight if overweight (even 5-10 lbs helps)',
                '🏃 Exercise regularly (30 minutes, most days)',
                '🚭 Quit smoking immediately',
                '🍷 Limit alcohol (max 1-2 drinks per day)',
                '☕ Reduce caffeine intake',
                '😴 Get 7-9 hours of quality sleep',
                '🧘 Practice stress management daily'
            ]
            
            recommendations['medical_advice'] = [
                'Monitor blood pressure at home twice daily',
                'Keep a blood pressure log',
                'Take medications exactly as prescribed',
                'Regular follow-ups every 2-4 weeks initially',
                'Watch for hypertension complications (kidney, eye, heart)',
                'Get regular kidney function tests',
                'Screen for secondary causes of hypertension',
                'Consider home blood pressure monitor'
            ]
            
            recommendations['prevention_tips'] = [
                'Learn proper blood pressure measurement technique',
                'Avoid high-sodium processed foods',
                'Increase potassium-rich foods (bananas, spinach)',
                'Eat dark chocolate (70%+ cocoa) in moderation',
                'Practice deep breathing exercises',
                'Avoid sudden position changes',
                'Monitor for symptoms: headaches, dizziness, chest pain'
            ]
            
        else:  # Low risk
            recommendations['risk_level'] = 'Low Risk'
            
            if trestbps >= 130:
                recommendations['immediate_actions'] = [
                    '⚠️ Your blood pressure is elevated - take preventive action now',
                    '📊 Start monitoring blood pressure regularly',
                    '🥗 Begin lifestyle modifications immediately',
                    '👨‍⚕️ Discuss with your doctor at next visit'
                ]
            else:
                recommendations['immediate_actions'] = [
                    '✅ Your blood pressure and hypertension risk are currently low',
                    '📋 Continue healthy lifestyle habits',
                    '🎯 Maintain regular health monitoring',
                    '💪 Keep up the good work!'
                ]
            
            recommendations['lifestyle_changes'] = [
                '🥗 Maintain balanced, low-sodium diet',
                '🏃 Continue regular physical activity',
                '⚖️ Maintain healthy body weight',
                '🚭 Stay tobacco-free',
                '🍷 Moderate alcohol consumption',
                '😴 Keep consistent sleep schedule',
                '🧘 Practice stress management',
                '💧 Stay well-hydrated'
            ]
            
            recommendations['medical_advice'] = [
                'Check blood pressure annually',
                'Monitor more frequently if family history exists',
                'Know your baseline blood pressure',
                'Discuss risk factors with your doctor',
                'Get cardiovascular screening as recommended'
            ]
            
            recommendations['prevention_tips'] = [
                'Keep sodium intake below 2300mg daily',
                'Eat plenty of fruits and vegetables',
                'Choose whole grains over refined grains',
                'Limit saturated and trans fats',
                'Stay physically active',
                'Maintain healthy weight',
                'Manage stress effectively'
            ]
        
        return recommendations

def get_feature_names():
    """Return feature names for hypertension prediction"""
    return ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']