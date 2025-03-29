from flask_mail import Message
from app import mail

def send_alert_email(recipient, prediction_data):
    """Send email alert for high-risk predictions"""
    disease = prediction_data.get('disease')
    location = prediction_data.get('location')
    risk = prediction_data.get('risk')
    cases = prediction_data.get('estimated_cases')
    
    subject = f"ALERT: {risk} Risk of {disease} Outbreak in {location}"
    
    body = f"""
    Dear Health Official,
    
    Our disease prediction system has identified a {risk} risk of {disease} outbreak in {location}, Malappuram.
    
    Prediction Details:
    - Disease: {disease}
    - Location: {location}
    - Risk Level: {risk}
    - Estimated Cases: {cases}
    
    Please take appropriate preventive measures and increase surveillance in the area.
    
    This is an automated alert from the Malappuram Disease Outbreak Prediction System.
    
    Regards,
    Disease Surveillance Team
    """
    
    msg = Message(
        subject=subject,
        recipients=[recipient],
        body=body
    )
    
    try:
        mail.send(msg)
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False 