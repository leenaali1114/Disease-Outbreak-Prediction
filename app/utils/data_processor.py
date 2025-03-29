import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
from datetime import datetime
from app.config import Config

class DataProcessor:
    def __init__(self):
        self.data_path = Config.DATA_PATH
        self.df = None
        self.preprocessor = None
        self.load_data()
        
    def load_data(self):
        """Load and preprocess the dataset"""
        self.df = pd.read_csv(self.data_path)
        
        # Convert date to datetime
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Extract month and season from date
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Season'] = self.df['Date'].dt.month.apply(self._get_season)
        
        # Filter out future dates for training
        self.train_df = self.df[self.df['Date'] <= datetime.now()]
        
        # Create preprocessor
        self._create_preprocessor()
        
    def _get_season(self, month):
        """Map month to season in Kerala context"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Summer'
        elif month in [6, 7, 8, 9]:
            return 'Monsoon'
        else:
            return 'Post-Monsoon'
    
    def _create_preprocessor(self):
        """Create a preprocessing pipeline for features"""
        # Categorical features
        categorical_features = ['Location', 'Disease', 'Population Density', 
                               'Water Quality Index', 'Air Quality Index', 
                               'Previous Outbreak History', 'Season']
        
        # Numerical features
        numerical_features = ['Rainfall (mm)', 'Temperature (°C)', 'Humidity (%)',
                             'Vaccination Rate (%)', 'Hospital Beds per 1000', 'Month']
        
        # Create transformers
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features),
                ('num', numerical_transformer, numerical_features)
            ])
        
        # Fit preprocessor on training data
        self.preprocessor.fit(self.train_df[categorical_features + numerical_features])
        
        # Save preprocessor
        os.makedirs(Config.MODEL_PATH, exist_ok=True)
        joblib.dump(self.preprocessor, os.path.join(Config.MODEL_PATH, 'preprocessor.pkl'))
    
    def preprocess_data(self, data):
        """Preprocess data for model input"""
        if isinstance(data, pd.DataFrame):
            # If data is a DataFrame
            if 'Date' in data.columns:
                data['Month'] = pd.to_datetime(data['Date']).dt.month
                data['Season'] = data['Month'].apply(self._get_season)
            
            categorical_features = ['Location', 'Disease', 'Population Density', 
                                   'Water Quality Index', 'Air Quality Index', 
                                   'Previous Outbreak History', 'Season']
            
            numerical_features = ['Rainfall (mm)', 'Temperature (°C)', 'Humidity (%)',
                                 'Vaccination Rate (%)', 'Hospital Beds per 1000', 'Month']
            
            return self.preprocessor.transform(data[categorical_features + numerical_features])
        else:
            # If data is a dictionary (from form input)
            input_df = pd.DataFrame([data])
            
            # Extract month and season from date
            if 'date' in data:
                date = pd.to_datetime(data['date'])
                input_df['Month'] = date.month
                input_df['Season'] = self._get_season(date.month)
            
            # Map column names
            column_mapping = {
                'location': 'Location',
                'disease': 'Disease',
                'rainfall': 'Rainfall (mm)',
                'temperature': 'Temperature (°C)',
                'humidity': 'Humidity (%)',
                'population_density': 'Population Density',
                'water_quality': 'Water Quality Index',
                'air_quality': 'Air Quality Index',
                'vaccination_rate': 'Vaccination Rate (%)',
                'hospital_beds': 'Hospital Beds per 1000',
                'previous_outbreak': 'Previous Outbreak History'
            }
            
            input_df.rename(columns=column_mapping, inplace=True)
            
            categorical_features = ['Location', 'Disease', 'Population Density', 
                                   'Water Quality Index', 'Air Quality Index', 
                                   'Previous Outbreak History', 'Season']
            
            numerical_features = ['Rainfall (mm)', 'Temperature (°C)', 'Humidity (%)',
                                 'Vaccination Rate (%)', 'Hospital Beds per 1000', 'Month']
            
            return self.preprocessor.transform(input_df[categorical_features + numerical_features])
    
    def get_locations(self):
        """Get unique locations in the dataset"""
        return sorted(self.df['Location'].unique())
    
    def get_diseases(self):
        """Get unique diseases in the dataset"""
        return sorted(self.df['Disease'].unique())
    
    def get_disease_stats(self):
        """Get statistics about diseases"""
        stats = {}
        for disease in self.get_diseases():
            disease_data = self.df[self.df['Disease'] == disease]
            stats[disease] = {
                'total_cases': disease_data['Cases'].sum(),
                'total_deaths': disease_data['Deaths'].sum(),
                'avg_cases': disease_data['Cases'].mean(),
                'max_cases': disease_data['Cases'].max(),
                'high_risk_count': len(disease_data[disease_data['Outbreak Risk'] == 'High']),
                'medium_risk_count': len(disease_data[disease_data['Outbreak Risk'] == 'Medium']),
                'low_risk_count': len(disease_data[disease_data['Outbreak Risk'] == 'Low'])
            }
        return stats
    
    def get_location_stats(self):
        """Get statistics about locations"""
        stats = {}
        for location in self.get_locations():
            location_data = self.df[self.df['Location'] == location]
            stats[location] = {
                'total_cases': location_data['Cases'].sum(),
                'total_deaths': location_data['Deaths'].sum(),
                'disease_count': location_data['Disease'].nunique(),
                'high_risk_count': len(location_data[location_data['Outbreak Risk'] == 'High']),
                'medium_risk_count': len(location_data[location_data['Outbreak Risk'] == 'Medium']),
                'low_risk_count': len(location_data[location_data['Outbreak Risk'] == 'Low'])
            }
        return stats
    
    def get_seasonal_patterns(self):
        """Get seasonal patterns of diseases"""
        seasonal_data = {}
        for disease in self.get_diseases():
            disease_df = self.df[self.df['Disease'] == disease]
            seasonal_counts = disease_df.groupby('Season')['Cases'].sum().to_dict()
            seasonal_data[disease] = seasonal_counts
        return seasonal_data
    
    def get_time_series_data(self):
        """Get time series data for visualization"""
        # Group by date and disease, sum cases
        time_series = self.df.groupby(['Date', 'Disease'])['Cases'].sum().reset_index()
        
        # Convert to format suitable for visualization
        result = {}
        for disease in self.get_diseases():
            disease_data = time_series[time_series['Disease'] == disease]
            result[disease] = {
                'dates': disease_data['Date'].dt.strftime('%Y-%m-%d').tolist(),
                'cases': disease_data['Cases'].tolist()
            }
        return result 