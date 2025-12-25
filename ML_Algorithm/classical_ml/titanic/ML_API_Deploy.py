from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize the App
app = FastAPI()

# Load the Model
model_path = "<absolute_path_to_model>/titanic_xgb_v1.pkl"
model = joblib.load('titanic_xgb_v1.pkl')

# Define the Input Format 
# Pydantic ensures the user sends the right data types.
class Passenger(BaseModel):
    pclass: int
    age: float
    sex: str
    fare: float
    sibsp: int

# Define the Endpoint 
# POST request because we are sending data TO the server
@app.post("/predict")
def prediction_survival(passenger: Passenger):
    # Convert incoming JSON to Pandas DataFrame
    data = {
        'pclass': [passenger.pclass],
        'sex': [passenger.sex],
        'age': [passenger.age],
        'fare': [passenger.fare],
        'sibsp': [passenger.sibsp]
    }
    df = pd.DataFrame(data)

    # We must repeat the exact same mapping we did in training
    df['sex'] = df['sex'].map({'male':0, 'female':1})

    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    # Return Result
    return {
        "survived": int(prediction),
        "survival_probability": float(probability)
    }
