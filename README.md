# Malappuram Disease Outbreak Prediction System

A Flask-based web application that uses machine learning to predict disease outbreaks in Malappuram district, Kerala, based on historical data and environmental factors. The system provides interactive visualizations, geographical mapping, and AI-powered preventive recommendations.

## Features

- **Disease Outbreak Prediction**: Predict the risk level, estimated cases, and deaths for various diseases in different locations
- **Interactive Dashboard**: Visualize disease patterns, seasonal trends, and historical data
- **Geographical Mapping**: View disease hotspots and risk areas across Malappuram district
- **AI-Powered Recommendations**: Get preventive measures using Groq's advanced LLMs
- **Email Alert System**: Automated notifications for high-risk predictions
- **Mobile-Responsive Design**: Accessible on various devices

## Technology Stack

- **Backend**: Flask (Python web framework)
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Data Visualization**: Chart.js, Leaflet.js
- **AI Integration**: Groq API for preventive recommendations
- **Email Service**: Flask-Mail for alert notifications

## Installation and Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/malappuram-disease-prediction.git
   cd malappuram-disease-prediction
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with the following variables:
   ```
   SECRET_KEY=your_secret_key
   GROQ_API_KEY=your_groq_api_key
   MAIL_SERVER=smtp.example.com
   MAIL_PORT=587
   MAIL_USE_TLS=True
   MAIL_USERNAME=your_email@example.com
   MAIL_PASSWORD=your_email_password
   MAIL_DEFAULT_SENDER=your_email@example.com
   ```

5. Run the application:
   ```
   python run.py
   ```

6. Access the application at `http://localhost:5000`

## Project Structure 