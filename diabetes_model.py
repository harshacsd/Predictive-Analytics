import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class DiabetesModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = 'ml_model/saved_models/diabetes_model.pkl'
        self.scaler_path = 'ml_model/saved_models/diabetes_scaler.pkl'
        
    def train(self, data_path='dataset/diabetes.csv'):
        """Train the diabetes prediction model"""
        # Load data
        df = pd.read_csv(data_path)
        
        # Prepare features and target
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
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
        
        if prediction == 1:  # High risk of diabetes
            recommendations['risk_level'] = 'High Risk'
            recommendations['immediate_actions'] = [
                '🚨 Consult with a doctor immediately for comprehensive diabetes screening',
                '🩺 Schedule an appointment with an endocrinologist',
                '📊 Get a complete blood sugar profile (HbA1c, Fasting, Post-prandial)',
                '👁️ Get your eyes checked for diabetic retinopathy risk'
            ]
            recommendations['lifestyle_changes'] = [
                '🥗 Adopt a low-glycemic index diet immediately',
                '🏃 Start moderate exercise (30 minutes daily walking)',
                '⚖️ Monitor and reduce body weight if BMI is high',
                '🚭 Quit smoking if you smoke',
                '💧 Stay well-hydrated (8-10 glasses of water daily)',
                '😴 Ensure 7-8 hours of quality sleep'
            ]
            recommendations['medical_advice'] = [
                'Monitor blood glucose levels regularly',
                'Consider diabetes management program',
                'Check for complications (kidney function, nerve damage)',
                'Discuss medication options with your doctor'
            ]
            recommendations['prevention_tips'] = [
                'Track your carbohydrate intake',
                'Avoid sugary drinks and processed foods',
                'Manage stress through meditation or yoga',
                'Regular health check-ups every 3 months'
            ]
        else:  # Low risk
            recommendations['risk_level'] = 'Low Risk'
            recommendations['immediate_actions'] = [
                '✅ Your current diabetes risk appears low',
                '📋 Continue regular health monitoring',
                '🎯 Maintain your current healthy lifestyle'
            ]
            recommendations['lifestyle_changes'] = [
                '🥗 Continue balanced diet with whole grains and vegetables',
                '🏃 Maintain regular physical activity (150 minutes/week)',
                '⚖️ Keep body weight in healthy range',
                '💧 Stay hydrated throughout the day',
                '😴 Maintain good sleep hygiene'
            ]
            recommendations['medical_advice'] = [
                'Get annual diabetes screening after age 45',
                'Monitor if you have family history of diabetes',
                'Regular check-ups with your primary care physician'
            ]
            recommendations['prevention_tips'] = [
                'Limit refined sugar and processed foods',
                'Include fiber-rich foods in diet',
                'Practice stress management',
                'Stay aware of diabetes symptoms'
            ]
        
        return recommendations

def get_feature_names():
    """Return feature names for diabetes prediction"""
    return ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']