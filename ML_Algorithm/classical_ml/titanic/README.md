# Titanic Survival Prediction (End-to-End ML Pipeline) 🚢

A production-ready project that trains an XGBoost model to predict Titanic survivors and serves it via a FastAPI microservice.

## Key Features
* **Algorithm:** XGBoost Classifier (Gradient Boosting).
* **Preprocessing:** Pandas pipeline with Median Imputation and One-Hot Encoding.
* **Tuning:** GridSearchCV for hyperparameter optimization.
* **Artifact:** Saves the trained model as `titanic_xgb_v1.pkl` for deployment.

## Accuracy
* **Test Accuracy:** ~86.19%

## 🛠️ Tech Stack
* **Training:** Scikit-Learn, XGBoost, Pandas
* **Serving:** FastAPI, Uvicorn, Pydantic
* **Serialization:** Joblib

## 📂 Project Structure
* `XGBoost_Titanic.py`: Data preprocessing, model training (XGBoost), and serialization.
* `ML_API_Deploy.py`: REST API to serve predictions in real-time.
* `titanic_xgb_v1.pkl`: The trained model artifact.

## 🚀 How to Run

### 1. Train the Model (Optional)
If you want to retrain the model from scratch:
```bash
python train_titanic.py
# This will generate titanic_xgb_v1.pkl
```

### 2. Start the API Server
Run the FastAPI server using Uvicorn:
```bash
uvicorn ML_API_Deploy:app --reload
```

### 3. Testing the API for predictions
Open your browser to http://127.0.0.1:8000/docs
```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)' \
  -H 'Content-Type: application/json' \
  -d '{
  "pclass": 1,
  "age": 25,
  "sex": "female",
  "fare": 100,
  "sibsp": 0
}'
```
Response should look like below
```
{
  "survived": 1,
  "survival_probability": 0.9353424906730652
}
```

> NOTE: Pre-requisites for app containerization: Verify docker is installed and service is up and running on the machine. 

## 🐳 Docker Support 
Running with Docker ensures the application works exactly the same on your machine as it does in production, regardless of OS or Python version.


### 1. Build the Docker Image
This packages the code, the model, and all dependencies (Linux + Python 3.9) into a portable image.
```bash
docker build -t titanic-api .
```

### 2. Run the Container
Start the container and map Port 80 of your machine to Port 80 of the container.
```bash
docker run -p 80:80 titanic-api
```

### 3. Verify Deployment
Once the container is running:
```bash
API Docs: Visit http://localhost/docs
Test Prediction: Use the Swagger UI or curl commands exactly as shown above.
```
