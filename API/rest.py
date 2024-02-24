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
    symptoms: str
    systemicManifestations: bool
    ANA: str
    Anti_dsDNA: str = ''
    RF: str
    CRP: str
    WBC: str
    RBC: str
    Hemoglobin: str
    Platelets: str = ''
    ESR: str
    FVC: str
    FEV1: str
    FEV1_FVC_Ratio: str = ''
    Creatinine: str = ''
    GFR: str
    C_Peptide: str
    Autoantibodies: str = ''
    Fasting_Glucose: str = ''
    HbA1c: str = ''
    Anti_CCP: str = ''
    Blood_Type: str = ''
    Blood_Pressure: str = ''
    Heart_Rate: str = ''
    Respiratory_Rate: str = ''
    Body_Temperature: str = ''
    Oxygen_Saturation: str = ''
    Cholesterol: str = ''
    ALT: str = ''
    AST: str = ''
    Current_Medications: str = ''
    X_ray_Findings: str = ''
    MRI_Findings: str = ''
    Echocardiogram_Results: str = ''
    
def standardize_feature_names(df):
    # Example transformation: Convert to uppercase and replace underscores with hyphens
    new_columns = {col: col.upper().replace('_', '-') for col in df.columns}
    return df.rename(columns=new_columns)

# Load JSON data
with open('../csvjson.json') as file:
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

@app.post("/predict/")
async def predict_diagnosis(record: MedicalRecord):
    input_data = pd.DataFrame([record.dict()])
    input_data = standardize_feature_names(input_data)

    # Now, apply any LabelEncoder transformations as before
    for column in input_data.columns:
        if column in label_encoders:
            le = label_encoders[column]
            input_data[column] = input_data[column].apply(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1)
    

    predicted_class_index = rf_Model.predict(input_data)[0]
    predicted_proba = rf_Model.predict_proba(input_data)[0]
    certainty_percentage = max(predicted_proba) * 100
    predicted_diagnosis = le_y.inverse_transform([predicted_class_index])[0]
    
    return {"Predicted Diagnosis": predicted_diagnosis, "Certainty": f"{certainty_percentage:.2f}%"}
