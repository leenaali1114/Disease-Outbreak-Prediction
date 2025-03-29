from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from app.models import disease_predictor
from app.utils.groq_service import GroqService
from app.utils.email_service import send_alert_email
import pandas as pd
from datetime import datetime

main_bp = Blueprint('main', __name__)
groq_service = GroqService()

@main_bp.route('/')
def index():
    """Home page"""
    locations = disease_predictor.data_processor.get_locations()
    diseases = disease_predictor.data_processor.get_diseases()
    return render_template('index.html', locations=locations, diseases=diseases)

@main_bp.route('/dashboard')
def dashboard():
    """Dashboard page with visualizations"""
    # Get statistics
    disease_stats = disease_predictor.data_processor.get_disease_stats()
    location_stats = disease_predictor.data_processor.get_location_stats()
    seasonal_patterns = disease_predictor.data_processor.get_seasonal_patterns()
    time_series_data = disease_predictor.data_processor.get_time_series_data()
    
    return render_template(
        'dashboard.html',
        disease_stats=disease_stats,
        location_stats=location_stats,
        seasonal_patterns=seasonal_patterns,
        time_series_data=time_series_data
    )

@main_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    locations = disease_predictor.data_processor.get_locations()
    diseases = disease_predictor.data_processor.get_diseases()
    
    if request.method == 'POST':
        # Get form data
        input_data = {
            'date': request.form.get('date'),
            'location': request.form.get('location'),
            'disease': request.form.get('disease'),
            'rainfall': float(request.form.get('rainfall')),
            'temperature': float(request.form.get('temperature')),
            'humidity': float(request.form.get('humidity')),
            'population_density': request.form.get('population_density'),
            'water_quality': request.form.get('water_quality'),
            'air_quality': request.form.get('air_quality'),
            'vaccination_rate': float(request.form.get('vaccination_rate')),
            'hospital_beds': float(request.form.get('hospital_beds')),
            'previous_outbreak': request.form.get('previous_outbreak')
        }
        
        # Make prediction
        prediction = disease_predictor.predict(input_data)
        
        # Prepare data for recommendations
        result_data = {
            'disease': input_data['disease'],
            'location': input_data['location'],
            'risk': prediction['risk'],
            'estimated_cases': prediction['estimated_cases']
        }
        
        # Get recommendations
        try:
            recommendations = groq_service.get_recommendations(result_data)
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            recommendations = """
            ## Preventive Recommendations
            
            ### For Local Government
            * Implement disease surveillance systems
            * Conduct public awareness campaigns
            * Ensure adequate medical supplies
            
            ### For Healthcare Facilities
            * Prepare isolation wards
            * Train staff on disease protocols
            * Stock necessary medications
            
            ### For General Public
            * Practice good hygiene
            * Seek medical attention for symptoms
            * Follow official health advisories
            """
        
        # Send email alert for high-risk predictions
        email = request.form.get('email')
        if email and prediction['risk'] == 'High':
            try:
                send_alert_email(email, result_data)
                flash('Alert email sent successfully!', 'success')
            except Exception as e:
                print(f"Error sending email: {e}")
                flash('Failed to send alert email.', 'danger')
        
        return render_template(
            'predict.html',
            locations=locations,
            diseases=diseases,
            input_data=input_data,
            prediction=prediction,
            recommendations=recommendations
        )
    
    return render_template('predict.html', locations=locations, diseases=diseases)

@main_bp.route('/map')
def map():
    """Map visualization page"""
    # Get location data for map
    location_stats = disease_predictor.data_processor.get_location_stats()
    
    # In a real application, you would have actual coordinates for each location
    # Here we're using placeholder coordinates for demonstration
    location_coordinates = {
        'Malappuram': [11.0509, 76.0710],
        'Manjeri': [11.1200, 76.1200],
        'Perinthalmanna': [10.9800, 76.2300],
        'Tirur': [10.9100, 75.9200],
        'Kottakkal': [11.0000, 76.0000],
        'Nilambur': [11.2700, 76.2300],
        'Ponnani': [10.7700, 75.9300],
        'Vengara': [11.0700, 76.0000],
        'Kondotty': [11.1400, 75.9600],
        'Valanchery': [10.8900, 76.0700],
        'Areekode': [11.1900, 76.0500],
        'Edappal': [10.7800, 76.0000],
        'Kalikavu': [11.2500, 76.3300],
        'Pandikkad': [11.1500, 76.1700],
        'Tanur': [10.8800, 75.8800],
        'Thiruvali': [11.1600, 76.2000],
        'Tirurangadi': [11.0400, 75.9400],
        'Chelembra': [11.0800, 76.0300],
        'Melattur': [11.0200, 76.1500],
        'Parappanangadi': [10.9300, 75.8700]
    }
    
    return render_template(
        'map.html',
        location_stats=location_stats,
        location_coordinates=location_coordinates
    )

@main_bp.route('/api/time-series')
def time_series_api():
    """API endpoint for time series data"""
    time_series_data = disease_predictor.data_processor.get_time_series_data()
    return jsonify(time_series_data)

@main_bp.route('/about')
def about():
    """About page"""
    return render_template('about.html') 