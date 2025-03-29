from app.models.disease_predictor import DiseasePredictor

disease_predictor = None

def init_models():
    global disease_predictor
    disease_predictor = DiseasePredictor()
    disease_predictor.load_or_train_models() 