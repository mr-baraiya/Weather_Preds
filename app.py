from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import List

app = FastAPI()

# Load the trained model, features, and scaler
model = joblib.load('weather_model.pkl')
model_features = joblib.load('model_features.pkl')
scaler = joblib.load('scaler.pkl')

class PredictionData(BaseModel):
    Temperature: float
    Humidity: int

@app.post("/predict")
def predict(data: List[PredictionData]):
    if model is None:
        raise HTTPException(status_code=400, detail="Model is not loaded. Please train the model first.")

    # Convert input data to DataFrame
    df = pd.DataFrame([d.dict() for d in data])

    # Align columns with training data
    X = df

    for col in model_features:
        if col not in X.columns:
            X[col] = 0
    X = X[model_features]

    # Scale the features
    X_scaled = scaler.transform(X)

    # Make predictions
    predictions = model.predict(X_scaled)

    return {"predictions": predictions.tolist()}

# Serve the static HTML file
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse('static/index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
