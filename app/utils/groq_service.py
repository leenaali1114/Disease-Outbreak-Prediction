import requests
import json
from app.config import Config

class GroqService:
    def __init__(self):
        self.api_key = Config.GROQ_API_KEY
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def get_recommendations(self, prediction_data):
        """Get preventive recommendations based on prediction data"""
        disease = prediction_data.get('disease')
        location = prediction_data.get('location')
        risk = prediction_data.get('risk')
        cases = prediction_data.get('estimated_cases')
        
        prompt = f"""
        As a public health expert, provide detailed preventive recommendations for a potential {disease} outbreak 
        in {location}, Malappuram district, Kerala, India. 
        
        The outbreak risk is predicted to be {risk} with an estimated {cases} cases.
        
        Format your response in clear markdown with the following structure:
        
        ## Preventive Recommendations for {disease} in {location}
        
        ### For Local Government
        * [Specific recommendation 1]
        * [Specific recommendation 2]
        * [Specific recommendation 3]
        
        ### For Healthcare Facilities
        * [Specific recommendation 1]
        * [Specific recommendation 2]
        * [Specific recommendation 3]
        
        ### For General Public
        * [Specific recommendation 1]
        * [Specific recommendation 2]
        * [Specific recommendation 3]
        
        ### Environmental Interventions
        * [Specific recommendation 1]
        * [Specific recommendation 2]
        
        ### Vaccination Recommendations
        * [Specific recommendation 1]
        * [Specific recommendation 2]
        
        Use bullet points for all recommendations and ensure the formatting is clean and readable.
        """
        
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "You are a public health expert specializing in infectious disease prevention and control in Kerala, India. You provide well-structured, clear recommendations in markdown format."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()
            
            result = response.json()
            recommendations = result['choices'][0]['message']['content']
            return recommendations
        
        except requests.exceptions.RequestException as e:
            print(f"Error calling Groq API: {e}")
            return """
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
            
            ### Environmental Interventions
            * Improve sanitation in affected areas
            * Control disease vectors if applicable
            
            ### Vaccination Recommendations
            * Follow standard vaccination protocols for the disease if available
            """ 