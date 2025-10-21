import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class HeartModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = 'ml_model/saved_models/heart_model.pkl'
        self.scaler_path = 'ml_model/saved_models/heart_scaler.pkl'
        
    def train(self, data_path='dataset/heart.csv'):
        """Train the heart disease prediction model"""
        # Load data
        df = pd.read_csv(data_path)
        
        # Prepare features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split data with stratification to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model with balanced class weights
        # This prevents bias towards the majority class
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',  # IMPORTANT: Balances predictions
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on test set
        test_accuracy = self.model.score(X_test_scaled, y_test)
        print(f"Model trained! Test accuracy: {test_accuracy:.3f}")
        
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
        """Get personalized recommendations based on prediction and heart rate analysis"""
        # Handle probability array - it should have shape (2,) with [prob_class_0, prob_class_1]
        if len(probability) > 1:
            risk_score = probability[1] * 100  # Probability of class 1 (high risk)
        else:
            risk_score = probability[0] * 100 if prediction == 1 else (1 - probability[0]) * 100
        
        # Extract heart rate (thalach) and other important metrics
        thalach = input_data[0][7]  # Maximum heart rate achieved
        age = input_data[0][0]
        exang = input_data[0][8]  # Exercise induced angina
        oldpeak = input_data[0][9]  # ST depression
        
        # Calculate expected max heart rate
        max_heart_rate = 220 - age
        heart_rate_percentage = (thalach / max_heart_rate) * 100
        
        recommendations = {
            'risk_level': '',
            'risk_score': risk_score,
            'heart_analysis': {},
            'immediate_actions': [],
            'lifestyle_changes': [],
            'medical_advice': [],
            'prevention_tips': []
        }
        
        # Heart rate analysis
        if thalach >= (max_heart_rate * 0.85):
            heart_status = 'Normal - Good cardiovascular fitness'
            breathing_status = 'Normal breathing capacity'
        elif thalach >= (max_heart_rate * 0.70):
            heart_status = 'Moderate - Regular monitoring recommended'
            breathing_status = 'Adequate breathing, could improve with exercise'
        else:
            heart_status = 'Below optimal - Needs attention'
            breathing_status = 'May indicate breathing difficulties or poor cardiovascular fitness'
        
        recommendations['heart_analysis'] = {
            'max_heart_rate': thalach,
            'expected_max': max_heart_rate,
            'percentage': heart_rate_percentage,
            'status': heart_status,
            'breathing': breathing_status
        }
        
        if prediction == 1:  # High risk of heart disease
            recommendations['risk_level'] = 'High Risk'
            
            # Check for breathing issues
            if exang == 1 or oldpeak > 2.0 or thalach < (max_heart_rate * 0.70):
                recommendations['immediate_actions'] = [
                    '🚨 URGENT: Consult a cardiologist immediately - you may have breathing difficulties',
                    '🏥 Visit emergency room if experiencing chest pain or severe shortness of breath',
                    '📞 Keep emergency contacts handy',
                    '🩺 Get ECG, Echocardiogram, and stress test done urgently',
                    '💊 Discuss immediate medication options with your doctor',
                    '🚑 Do NOT ignore symptoms like breathing problems, chest discomfort, or fatigue'
                ]
            else:
                recommendations['immediate_actions'] = [
                    '🚨 Schedule urgent appointment with cardiologist',
                    '🩺 Get comprehensive cardiac evaluation (ECG, Echo, Stress Test)',
                    '📊 Complete lipid profile and cardiac markers test',
                    '💊 Discuss preventive medication with your doctor'
                ]
            
            recommendations['lifestyle_changes'] = [
                '🥗 Start heart-healthy diet (Mediterranean or DASH diet)',
                '🧂 Reduce sodium intake to less than 2000mg per day',
                '🏃 Begin supervised cardiac rehabilitation program',
                '🚭 Quit smoking immediately if you smoke',
                '🍷 Limit alcohol consumption',
                '⚖️ Work on achieving healthy weight',
                '😴 Ensure 7-8 hours quality sleep',
                '🧘 Practice stress reduction (meditation, deep breathing)'
            ]
            
            recommendations['medical_advice'] = [
                'Consider aspirin therapy (consult doctor first)',
                'Monitor blood pressure daily',
                'Get cholesterol levels checked regularly',
                'Discuss statin therapy if cholesterol is high',
                'Regular follow-ups every 2-4 weeks initially',
                'Consider cardiac CT or angiography if recommended'
            ]
            
            recommendations['prevention_tips'] = [
                'Learn CPR and warning signs of heart attack',
                'Keep nitroglycerin available if prescribed',
                'Avoid strenuous activities until cleared by doctor',
                'Track symptoms daily (chest pain, breathing, fatigue)',
                'Monitor heart rate and blood pressure',
                'Avoid extreme temperatures'
            ]
            
        else:  # Low risk
            recommendations['risk_level'] = 'Low Risk'
            
            if thalach >= (max_heart_rate * 0.85):
                recommendations['immediate_actions'] = [
                    '✅ Excellent! Your heart rate indicates good cardiovascular fitness',
                    '🎯 Continue your current healthy lifestyle',
                    '📋 Regular annual check-ups recommended',
                    '💪 Maintain your exercise routine for optimal heart health'
                ]
            else:
                recommendations['immediate_actions'] = [
                    '✅ Your heart disease risk is currently low',
                    '📊 Your heart rate could be improved with regular exercise',
                    '🏃 Gradually increase cardiovascular exercise intensity',
                    '📋 Continue regular health monitoring'
                ]
            
            recommendations['lifestyle_changes'] = [
                '🥗 Maintain balanced diet rich in fruits, vegetables, whole grains',
                '🏃 Continue regular aerobic exercise (150 minutes/week)',
                '🧂 Keep sodium intake moderate',
                '⚖️ Maintain healthy weight',
                '💧 Stay well-hydrated',
                '😴 Maintain consistent sleep schedule',
                '🧘 Practice stress management techniques'
            ]
            
            recommendations['medical_advice'] = [
                'Annual cardiac check-up after age 40',
                'Monitor blood pressure and cholesterol yearly',
                'Discuss family history with your doctor',
                'Get baseline cardiac tests as recommended'
            ]
            
            recommendations['prevention_tips'] = [
                'Continue heart-healthy habits',
                'Stay aware of cardiac symptoms',
                'Avoid smoking and excessive alcohol',
                'Manage stress effectively',
                'Stay physically active',
                'Monitor changes in energy levels or breathing'
            ]
        
        return recommendations

def get_feature_names():
    """Return feature names for heart disease prediction"""
    return ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']