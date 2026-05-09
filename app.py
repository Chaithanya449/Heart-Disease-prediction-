from fastapi import FastAPI
from pydantic import BaseModel,Field
import pickle 
import numpy as np 

app  = FastAPI(title = 'Heart Disease Prediction API',description = 'API for predicting heart disease based on patient data')
# load model and scaler
model = pickle.load(open('models/best_model.pkl','rb'))
scaler = pickle.load(open('models/scaler.pkl','rb'))

class PatientData(BaseModel):
    age: float
    trestbps: float
    chol: float
    fbs: float
    thalch: float
    oldpeak: float
    sex_Male: int
    cp_atypical_angina: int = Field(alias='cp_atypical angina')
    cp_non_anginal: int     = Field(alias='cp_non-anginal')
    cp_typical_angina: int  = Field(alias='cp_typical angina')
    restecg_normal: int
    restecg_st_t_abnormality: int = Field(alias='restecg_st-t abnormality')
    exang_TRUE: int
    exang_TURE: int 
    slope_flat: int
    slope_upsloping: int
    thal_normal: int
    thal_reversable_defect: int = Field(alias='thal_reversable defect')

    model_config = {'populate_by_name': True}
@app.get('/')
def home():
    return{'message':'Heart Disease Prediction API is running!'}
@app.post('/predict')
def predict(input_data: PatientData):
    features = np.array([[
        input_data.age,
        input_data.trestbps,
        input_data.chol,
        input_data.fbs,
        input_data.thalch,
        input_data.oldpeak,
        input_data.sex_Male,
        input_data.cp_atypical_angina,
        input_data.cp_non_anginal,
        input_data.cp_typical_angina,
        input_data.restecg_normal,
        input_data.restecg_st_t_abnormality,
        input_data.exang_TRUE,
        input_data.exang_TURE,
        input_data.slope_flat,
        input_data.slope_upsloping,
        input_data.thal_normal,
        input_data.thal_reversable_defect

]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]
    return {
        'Prediction': int(prediction),
        'Result': 'Heart disease is detected' if prediction == 1 else 'No heart disease detected',
        'Probability': f'{probability:.2f}'
    }