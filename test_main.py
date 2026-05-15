from fastapi.testclient import TestClient
from app import app 

client = TestClient(app)

# Sample valid input — from your actual test set
valid_input = {
    "age": 61,
    "trestbps": 150,
    "chol": 243,
    "fbs": 1,
    "thalch": 137,
    "oldpeak": 1.0,
    "sex_Male": 1,
    "cp_atypical angina": 0,
    "cp_non-anginal": 1,
    "cp_typical angina": 0,
    "restecg_normal": 1,
    "restecg_st-t abnormality": 0,
    "exang_TRUE": 1,
    "exang_TURE": 0,
    "slope_flat": 1,
    "slope_upsloping": 0,
    "thal_normal": 1,
    "thal_reversable defect": 0
}

# Test 1 to check API is running
def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {'message':'Heart Disease Prediction API is running!'}

# Test 2: Test the heart disease prediction endpoint with valid input
def test_predict_valid_input():
    response = client.post('/predict', json = valid_input)
    assert response.status_code == 200
    data  = response.json()
    assert "Prediction" in data
    assert "Result" in data 
    assert "Probability" in data
    assert data["Prediction"] in [0,1]

# Test 3: Test for invalid input 
def testa_predict_invalid_input():
    response = client.post('/predict',json = {"age": "invalid"})
    assert response.status_code == 422
    