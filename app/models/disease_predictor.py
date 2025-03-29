import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from app.utils.data_processor import DataProcessor
from app.config import Config

class DiseasePredictor:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.risk_model = None
        self.cases_model = None
        self.deaths_model = None
        self.model_path = Config.MODEL_PATH
        
    def load_or_train_models(self):
        """Load trained models if they exist, otherwise train new ones"""
        risk_model_path = os.path.join(self.model_path, 'risk_model.pkl')
        cases_model_path = os.path.join(self.model_path, 'cases_model.pkl')
        deaths_model_path = os.path.join(self.model_path, 'deaths_model.pkl')
        
        if (os.path.exists(risk_model_path) and 
            os.path.exists(cases_model_path) and 
            os.path.exists(deaths_model_path)):
            # Load existing models
            self.risk_model = joblib.load(risk_model_path)
            self.cases_model = joblib.load(cases_model_path)
            self.deaths_model = joblib.load(deaths_model_path)
            print("Loaded existing models")
        else:
            # Train new models
            self.train_models()
            
    def train_models(self):
        """Train prediction models"""
        print("Training new models...")
        
        # Get preprocessed data
        df = self.data_processor.train_df
        
        # Features for prediction
        features = ['Location', 'Disease', 'Rainfall (mm)', 'Temperature (°C)', 
                   'Humidity (%)', 'Population Density', 'Water Quality Index', 
                   'Air Quality Index', 'Vaccination Rate (%)', 'Hospital Beds per 1000',
                   'Previous Outbreak History', 'Month', 'Season']
        
        # Preprocess data
        X = self.data_processor.preprocess_data(df[features])
        
        # Train risk prediction model (classification)
        y_risk = df['Outbreak Risk']
        X_train, X_test, y_train, y_test = train_test_split(X, y_risk, test_size=0.2, random_state=42)
        
        self.risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.risk_model.fit(X_train, y_train)
        
        # Evaluate risk model
        y_pred = self.risk_model.predict(X_test)
        risk_accuracy = accuracy_score(y_test, y_pred)
        print(f"Risk Model Accuracy: {risk_accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Train cases prediction model (regression)
        y_cases = df['Cases']
        X_train, X_test, y_train, y_test = train_test_split(X, y_cases, test_size=0.2, random_state=42)
        
        self.cases_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.cases_model.fit(X_train, y_train)
        
        # Evaluate cases model
        y_pred = self.cases_model.predict(X_test)
        cases_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Cases Model RMSE: {cases_rmse:.4f}")
        
        # Train deaths prediction model (regression)
        y_deaths = df['Deaths']
        X_train, X_test, y_train, y_test = train_test_split(X, y_deaths, test_size=0.2, random_state=42)
        
        self.deaths_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.deaths_model.fit(X_train, y_train)
        
        # Evaluate deaths model
        y_pred = self.deaths_model.predict(X_test)
        deaths_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Deaths Model RMSE: {deaths_rmse:.4f}")
        
        # Save models
        os.makedirs(self.model_path, exist_ok=True)
        joblib.dump(self.risk_model, os.path.join(self.model_path, 'risk_model.pkl'))
        joblib.dump(self.cases_model, os.path.join(self.model_path, 'cases_model.pkl'))
        joblib.dump(self.deaths_model, os.path.join(self.model_path, 'deaths_model.pkl'))
        
    def predict(self, input_data):
        """Make predictions using trained models"""
        # Preprocess input data
        X = self.data_processor.preprocess_data(input_data)
        
        # Make predictions
        risk = self.risk_model.predict(X)[0]
        cases = max(0, round(self.cases_model.predict(X)[0]))
        deaths = max(0, round(self.deaths_model.predict(X)[0]))
        
        # Get risk probability
        risk_proba = self.risk_model.predict_proba(X)[0]
        risk_proba_dict = {
            'Low': risk_proba[list(self.risk_model.classes_).index('Low')] if 'Low' in self.risk_model.classes_ else 0,
            'Medium': risk_proba[list(self.risk_model.classes_).index('Medium')] if 'Medium' in self.risk_model.classes_ else 0,
            'High': risk_proba[list(self.risk_model.classes_).index('High')] if 'High' in self.risk_model.classes_ else 0
        }
        
        return {
            'risk': risk,
            'risk_probability': risk_proba_dict,
            'estimated_cases': int(cases),
            'estimated_deaths': int(deaths)
        } 