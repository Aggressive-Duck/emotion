from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
import os
import preprocessing

app = FastAPI()

# Mount static files (frontend)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Global variables for model artifacts
model_artifacts = {}

@app.on_event("startup")
def load_model():
    """
    Load the pre-trained model and preprocessing artifacts on startup.
    """
    model_dir = "./models"
    try:
        model_artifacts['vectorizer'] = joblib.load(os.path.join(model_dir, 'vectorizer.joblib'))
        model_artifacts['scaler'] = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        model_artifacts['pca'] = joblib.load(os.path.join(model_dir, 'pca.joblib'))
        model_artifacts['label_encoder'] = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
        model_artifacts['svm_model'] = joblib.load(os.path.join(model_dir, 'svm_model.joblib'))
        model_artifacts['metadata'] = joblib.load(os.path.join(model_dir, 'metadata.joblib'))
        print("All model artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        raise e

class EmailRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_emotion(request: EmailRequest):
    try:
        text = request.text
        
        # 1. Preprocess
        # We need to pass the frequent and rare words loaded from training
        frequent_words = model_artifacts['metadata']['frequent_words']
        rare_words = model_artifacts['metadata']['rare_words']
        
        processed_text = preprocessing.full_preprocess_pipeline(text, frequent_words, rare_words)
        
        # 2. Vectorize
        # Note: transform expects an iterable (list), so we wrap text in []
        vectorized_text = model_artifacts['vectorizer'].transform([processed_text]).toarray()
        
        # 3. Scale
        scaled_text = model_artifacts['scaler'].transform(vectorized_text)
        
        # 4. PCA
        pca_text = model_artifacts['pca'].transform(scaled_text)
        
        # 5. Predict
        prediction_index = model_artifacts['svm_model'].predict(pca_text)
        
        # 6. Decode Label
        prediction_label = model_artifacts['label_encoder'].inverse_transform(prediction_index)[0]
        
        return {"emotion": prediction_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Go to /static/index.html to use the app"}
