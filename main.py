import uvicorn
from os import getenv
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, HTTPException, Path
from starlette.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load the model from the .pkl file
loaded_knn = None
with open('knn_classifier.pkl', 'rb') as f:
    loaded_knn = pickle.load(f)


def classify_and_find_farthest(x, y, z):
    # Create a DataFrame for the new points
    new_points = pd.DataFrame({
        'Food': [x, y, z],
        'Healthcare': [y, z, x],
        'Fashion': [z, x, y]
    })

    # Classify the new points using the loaded KNN classifier
    predicted_labels = loaded_knn.predict(new_points)

    # Find the cluster centroids
    centroids = []
    for label in predicted_labels:
        points_with_label = new_points[predicted_labels == label]
        centroid = np.mean(points_with_label, axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)
    
    # Find the farthest point from its respective cluster
    farthest_point = None
    max_distance = float('-inf')

    for point, label in zip(new_points.values, predicted_labels):
        centroid = centroids.values
        distance = np.linalg.norm(point - centroid)
        if distance > max_distance:
            max_distance = distance
            farthest_point = point

    return farthest_point

# Initialize Firebase Admin SDK
credentials_firebase = {
    "type": "service_account",
    "project_id": "hackmatrix-ba8fb",
    "private_key_id": "c3bfdfb17f35146f36f52a0e105058a5c4f43c69",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQCrfl9CewDW42pC\nr9+J26ND/HpNy4/F5B1umJu/o+8di+N7k+q1GN/4gYF+cfR1bRnO2VWM9j3Fc0WS\nZ35C6SGV2xdyjjAco4ioopV6C5qYcZwAyZP/p0iRjb6LHgDMCnPdCpGSF8XFATQx\njJ1vG9vu7fFbrrVqVgHzTnUp6bxCsMIvi2fDGZnrRdPCC+ufow4KJgbTQtDqWKyD\nDdrTQzbE2SjRUjOnLDTxABZfDcGtqM5AjuQ68w9Cguu/aPDKi9gY22ITNreX29g5\n2WZfUrHYOw25F6V6XrFIOuDf2/3jEXvy5cIgzqGj+NQ87SaQwm/XyEsv7hpTZMW6\nwqnQjGZdAgMBAAECggEABtwbuF+ONYpMPlWlpfCMs9P+Gm2JlztcOAfPtxc6Id7u\nHwvYKB1SHDEl+mZZhnbyQNVtuCFDcEn/nu8X3FpR9xoX8oOghgy+kyxJuWOMcAuf\nV2K40lGhM/1NAWiWVJMYdl+NWiAsT4iQS2kaBQ9CuWh4Lpgq9pFxK4fYZPBOEQbL\nQcPybGENy38+Nf9DrgfjUS1y8bbEI5XBozJ8RVC1jzX8Lef0njtJhe2mZjNQMvFp\nHc1FoTFAMjGi8+apO266ja4YYjjzQwSsemLD0pvllyQHO+xOy3XKhPNF+Tg3DRs3\nAXYoutETR4FwHAPTolzo3P6fgjyBYvqDyKfW3tlvkQKBgQDVVxVvTpJPrnIInKRL\nB9F7KLPmRNZV+JuA0WiGfej2OPrzHz9KI4GSiA6DBeBpDbZduEy7jNmTxlGS4SN3\nar09R4HM3pTZLetYTfa0SxjMTAMmbzKBmyKC/M7Vv3P17+bmTsXLSMrjZIRYH+8e\nRwEfpKTbvOFeArZJGG8jyEowZQKBgQDNySh2sKMBuXqdeCtChf177PFoRG4nnHLJ\nEWIyBwk2J20vmlSuBlY8iScO2LFJXaEWhfl2UYkDnsdUUtvyKcJmPXA7erccLt11\nH/XQwTLlGsgaEtBqjB+FeeB0WXd7lvR5KGGRrTZeDP7ZbR+o3JLbUTeZv1DvV0hp\nrC1jGj7ymQKBgCGUz57xsz4vq2uHnKTi2iqUwZyhgUuPEos4a0egUidP2NCkPoYh\nCKhUGlStfCGNMwOVmx56kVUdhoGkRrzpZFhdBSWGc8+r1rvTqd2/ZGvkGyrVnhGg\npdIQkU48ELjJxoLCK4hQMP+SNvLYM/+EFb0xYXHlTWRK8P6YhgYP5P2xAoGAQdu0\n9XdGU9D2atsAjUOwgi6se8AauNaa7bqAgJ471nb7vJZZr3AbvTfvphK3elFasoih\n87nYba4tANGbzn6K1omnF4IIhB6DhW57DxolnajajW2kAdViaSc+LD5NvOHsz7Ga\nuDKFCciC7za7QSGGZmYxsyTFVDPM2vTdea/2oVECgYBVfhStujfj/ZJyVz2WHLm1\nmAeMrBLShi7iaAXHwkMdzrkq4tGnLwkMAWbzTur/9QC6exG+bm92TVqUmyK9qlkY\nGusPu94aqemac6uIEf/4z26y3sgPwFJ34fuihsBGeTOwfjCfZWzNuXtMj/f/QO4d\nPwXtGHhRIFQOr7TzXD19Ig==\n-----END PRIVATE KEY-----\n",
    "client_email": "firebase-adminsdk-aew9a@hackmatrix-ba8fb.iam.gserviceaccount.com",
    "client_id": "113847620523291284487",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-aew9a%40hackmatrix-ba8fb.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com"
}

# Initialize Firebase Admin SDK
cred = credentials.Certificate(credentials_firebase)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Define the collection name
collection_name = "CategorizeExpense"

# Create the FastAPI instance
app = FastAPI()

# Define CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class UserInput(BaseModel):
    user_id: str

class PredictionResult(BaseModel):
    forecast: List[float]

def fetch_expenses(user_id):
    # Fetch input data from Firebase
    emp_ref = db.collection(collection_name).where('UserId', '==', user_id)
    docsm = emp_ref.stream()

    for doc in docsm:
        # Get the document ID
        doc_id = doc.id
        
        # Get the data from the document
        data = doc.to_dict()

        # Check if MonthlyExpenses exist
        if 'MonthlyCategory' in data:
            monthly_expenses = data['MonthlyCategory']

            # Check if Feb exists
            if 'Feb' in monthly_expenses:
                feb_expenses = monthly_expenses['Feb']

                # Check if Fashion and Food exist
                if 'Fashion' in feb_expenses and 'Food' in feb_expenses and 'Health' in feb_expenses:
                    fashion_expense = feb_expenses['Fashion']
                    print(fashion_expense)
                    food_expense = feb_expenses['Food']
                    print(food_expense)
                    health_expense = feb_expenses['Health']
                    print(health_expense)

                    # Return the document ID and expenses
                    return food_expense, health_expense, fashion_expense
                else:
                    return None, None, None
            else:
                return None, None, None
        else:
            return None, None, None

    # If no document found for the user ID
    return None, None, None

@app.get("/")
def read_root():
    return {"message": "Welcome to the Expense Forecasting API!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/predict/{user_id}", response_model=PredictionResult)
async def predict(user_id: str = Path(..., title="The ID of the user to predict expenses for")):
    try:
        # Fetch input data from Firebase
        x, y, z = fetch_expenses(user_id)

        forecast_dataval = classify_and_find_farthest(x, y, z)
        
        # Return prediction results as JSON
        return {"Distance Sequence": forecast_dataval.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(getenv("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
