import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

class StrokeModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model_path = 'ml_model/saved_models/stroke_model.pkl'
        self.scaler_path = 'ml_model/saved_models/stroke_scaler.pkl'
        self.encoders_path = 'ml_model/saved_models/stroke_encoders.pkl'
        
    def train(self, data_path='dataset/stroke.csv'):
        """Train the stroke prediction model"""
        # Load data
        df = pd.read_csv(data_path)
        
        # Drop id column
        df = df.drop('id', axis=1)
        
        # Encode categorical variables
        categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        # Prepare features and target
        X = df.drop('stroke', axis=1)
        y = df['stroke']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.model.fit(X_train_scaled, y_train)
        
        # Save model, scaler, and encoders
        os.makedirs('ml_model/saved_models', exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.label_encoders, self.encoders_path)
        
        return self.model
    
    def load_model(self):
        """Load trained model, scaler, and encoders"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path) and os.path.exists(self.encoders_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.label_encoders = joblib.load(self.encoders_path)
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
        
        if prediction == 1:  # High risk of stroke
            recommendations['risk_level'] = 'High Risk'
            recommendations['immediate_actions'] = [
                '🚨 Seek immediate medical consultation with a neurologist',
                '🏥 Get brain MRI/CT scan to assess current brain health',
                '🩺 Complete cardiovascular and neurological examination',
                '📊 Check blood pressure, cholesterol, and blood sugar levels',
                '💊 Discuss antiplatelet or anticoagulant therapy with doctor',
                '🚑 Learn F.A.S.T. stroke warning signs (Face, Arms, Speech, Time)'
            ]
            recommendations['lifestyle_changes'] = [
                '🚭 Quit smoking immediately - major stroke risk factor',
                '🍷 Eliminate or severely limit alcohol consumption',
                '🥗 Adopt DASH or Mediterranean diet',
                '🧂 Reduce sodium to less than 1500mg per day',
                '⚖️ Achieve and maintain healthy weight',
                '🏃 Start supervised exercise program (30 min, 5 days/week)',
                '😴 Ensure 7-8 hours of quality sleep',
                '🧘 Practice stress reduction techniques daily'
            ]
            recommendations['medical_advice'] = [
                'Monitor blood pressure at least twice daily',
                'Control cholesterol with medication if needed',
                'Manage diabetes strictly if diabetic',
                'Consider carotid artery screening',
                'Regular neurological assessments',
                'Discuss aspirin therapy or other blood thinners',
                'Get cardiac evaluation to rule out atrial fibrillation',
                'Follow up every 1-2 months initially'
            ]
            recommendations['prevention_tips'] = [
                'Learn and memorize stroke warning signs: F.A.S.T.',
                'Keep emergency contacts readily available',
                'Avoid strenuous activities until cleared by doctor',
                'Monitor for symptoms: sudden numbness, confusion, vision problems',
                'Stay compliant with all prescribed medications',
                'Avoid extreme temperature changes',
                'Consider medical alert device'
            ]
        else:  # Low risk
            recommendations['risk_level'] = 'Low Risk'
            recommendations['immediate_actions'] = [
                '✅ Your stroke risk appears low currently',
                '📋 Maintain regular health check-ups',
                '🎯 Continue healthy lifestyle practices',
                '📚 Stay informed about stroke prevention'
            ]
            recommendations['lifestyle_changes'] = [
                '🥗 Maintain heart-healthy diet',
                '🏃 Continue regular physical activity (150 min/week)',
                '🚭 Stay tobacco-free',
                '🍷 Limit alcohol consumption',
                '⚖️ Maintain healthy weight',
                '💧 Stay well-hydrated',
                '😴 Keep consistent sleep schedule',
                '🧘 Manage stress effectively'
            ]
            recommendations['medical_advice'] = [
                'Annual health screenings after age 55',
                'Monitor blood pressure regularly',
                'Check cholesterol levels yearly',
                'Control blood sugar if diabetic or prediabetic',
                'Discuss family history with your doctor',
                'Get baseline cardiovascular assessment'
            ]
            recommendations['prevention_tips'] = [
                'Learn F.A.S.T. stroke warning signs',
                'Control blood pressure (keep below 120/80)',
                'Stay physically and mentally active',
                'Limit saturated fats and trans fats',
                'Eat foods rich in potassium and fiber',
                'Be aware of stroke symptoms in your family history'
            ]
        
        return recommendations

def get_feature_names():
    """Return feature names for stroke prediction"""
    return ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
            'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']