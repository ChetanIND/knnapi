from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import numpy as np
import h5py
import os

# Define the FastAPI app
app = FastAPI()

# Define the input data model
class InputData(BaseModel):
    Food: float
    Healthcare: float
    Fashion: float

# Define the output data model
class PredictionResult(BaseModel):
    predicted_label: int

def load_model():
    model_path = "model.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found.")
    
    with h5py.File(model_path, 'r') as file:
        model = file['model'][()]  # Load the single value
        
    return model


loaded_model = load_model()

# Define the endpoint
@app.get("/classify/", response_model=PredictionResult)
async def classify(Food: float = Query(..., description="Value of Food field"),
                   Healthcare: float = Query(..., description="Value of Healthcare field"),
                   Fashion: float = Query(..., description="Value of Fashion field")):
    
    # Classify the new point using the loaded model
    predicted_label = loaded_model.predict(np.array([[Food, Healthcare, Fashion]]))
    
    return {"predicted_label": predicted_label[0]}

# Example usage: send a GET request to http://127.0.0.1:8000/classify/?Food=3500&Healthcare=700&Fashion=450
