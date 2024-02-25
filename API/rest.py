from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

class MedicalRecord(BaseModel):
    gender: str
    age: int
    blood_type: str
    symptoms: str
    systemicManifestations: bool
    ANA: str
    RF : str
    CRP: str
    WBC: str
    RBC: str
    Hemoglobin: str
    Platelets: str
    ESR: str
    FVC: str
    FEV1: str
    Creatinine: str
    GFR: str
    Autoantibodies: str
    HbA1c: str
    vital_signs_blood_pressure: str
    vital_signs_heart_rate: int
    additional_blood_tests_lipid_profile_HDL: str
    imaging_and_diagnostic_tests: str
    medication_and_treatment_history: str
    ANTI_DSDNA: str
    ANTI_CCP: str
    C_PEPTIDE: str
    FASTING_GLUCOSE: str
    FEV_FVC_RATIO: str
    
        
def standardize_feature_names(df):
    # Example transformation: Convert to uppercase and replace underscores with hyphens
    new_columns = {col: col.upper().replace('_', '-') for col in df.columns}
    return df.rename(columns=new_columns)

# Load JSON data
with open('csvjson5K.json') as file:
    print("Loading JSON data...")
    data = json.load(file)
df = pd.json_normalize(data)
df = standardize_feature_names(df)

# Data preprocessing (as per your original code)
label_encoders = {}
for column in df.columns:
    if df[column].apply(lambda x: isinstance(x, list)).any():
        df[column] = df[column].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else x)
    if column != 'FINALDIAGNOSIS' and df[column].dtype == 'object':
        df[column] = df[column].astype(str)
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

df = standardize_feature_names(df)
X = df.drop('FINALDIAGNOSIS', axis=1)  # Use the correct column name as per your standardization
y = df['FINALDIAGNOSIS']
le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

rf_Model = RandomForestClassifier(n_estimators=100, random_state=64)
rf_Model.fit(X_train, y_train)

@app.post("/predict")
async def predict_diagnosis(record: MedicalRecord):
    # predict the diagnosis
    data = {
        "GENDER": [record.gender],
        "AGE": [record.age],
        "BLOOD-TYPE": [record.blood_type],
        "SYMPTOMS": [record.symptoms],
        "SYSTEMIC-MANIFESTATIONS": [record.systemicManifestations],
        "ANA": [record.ANA],
        "RF": [record.RF],
        "CRP": [record.CRP],
        "WBC": [record.WBC],
        "RBC": [record.RBC],
        "HEMOGLOBIN": [record.Hemoglobin],
        "PLATELETS": [record.Platelets],
        "ESR": [record.ESR],
        "FVC": [record.FVC],
        "FEV1": [record.FEV1],
        "CREATININE": [record.Creatinine],
        "GFR": [record.GFR],
        "AUTOANTIBODIES": [record.Autoantibodies],
        "HBA1C": [record.HbA1c],
        "VITAL-SIGNS-BLOOD-PRESSURE": [record.vital_signs_blood_pressure],
        "VITAL-SIGNS-HEART-RATE": [record.vital_signs_heart_rate],
        "ADDITIONAL-BLOOD-TESTS-LIPID-PROFILE-HDL": [record.additional_blood_tests_lipid_profile_HDL],
        "IMAGING-AND-DIAGNOSTIC-TESTS": [record.imaging_and_diagnostic_tests],
        "MEDICATION-AND-TREATMENT-HISTORY": [record.medication_and_treatment_history],
        "ANTI-DSDNA": [record.ANTI_DSDNA],
        "ANTI-CCP": [record.ANTI_CCP],
        "C-PEPTIDE": [record.C_PEPTIDE],
        "FASTING-GLUCOSE": [record.FASTING_GLUCOSE],
        "FEV-FVC-RATIO": [record.FEV_FVC_RATIO]
    }
    df = pd.DataFrame(data)
    df = standardize_feature_names(df)
    for column in df.columns:
        if column in label_encoders:
            df[column] = label_encoders[column].transform(df[column])
    prediction = rf_Model.predict(df)
    return {"diagnosis": le_y.inverse_transform(prediction)[0]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)