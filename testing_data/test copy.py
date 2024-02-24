<<<<<<< HEAD
<<<<<<< HEAD
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder



encoder = OneHotEncoder()


# Load JSON data
with open('csvjson(2).json') as file:
    data = json.load(file)

# Normalize JSON data into a flat table
df = pd.json_normalize(data)



# Automatically handle columns with lists
for column in df.columns:
    if df[column].apply(lambda x: isinstance(x, list)).any():
        df[column] = df[column].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else x)


# Encode categorical variables
label_encoders = {}
# Convert columns to strings before encoding
for column in df.select_dtypes(include=['object']).columns:
    if column != 'finalDiagnosis':  # Assuming 'finalDiagnosis' is the target variable
        # Convert the column to string type
        df[column] = df[column].astype(str)  # Ensure all data is string type
        
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        
        # print what each label corresponds to
        
        # print(column, le.classes_)
        
        label_encoders[column] = le

        


# Split data into features and target variable
X = df.drop('finalDiagnosis', axis=1)  # Replace 'finalDiagnosis' with your actual target column name if different
y = df['finalDiagnosis']  # Replace 'finalDiagnosis' with your actual target column name if different

# Optionally, encode the target variable if it's categorical
le_y = LabelEncoder()
# print the values of the target variable
y_encoded = le_y.fit_transform(y)
# print("Target Variable Values:", le_y.classes_)

# Split data into training and test sets

# print results
# print(X_train.head())
# print(X_train.info())
# print(X_train.describe())
# print("null:", X_train.isnull().sum())
# print("X Train Shape:", X_train.shape)
# print("X Test Shape:", X_test.shape)
# print("Y Train Shape:", y_train.shape)
# print("Y Test Shape:", y_test.shape)

##########################################################

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

rf_Model = RandomForestClassifier(n_estimators=100, random_state=64)
rf_Model.fit(X_train, y_train)
rf_Model.score(X_test, y_test)
y_pred = rf_Model.predict(X_test)

# print out the accuracy
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print out the confusion matrix
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print out the confusion matrix
# print("Classification Report:", classification_report(y_test, y_pred))

# # compare model predictions with actual values
# print("Predictions:", y_pred)
# print("Actual:     ", y_test)

# # print out the accuracy
# print("Accuracy:", accuracy_score(y_test, y_pred))


# test the model with specific data

Test_Medical_Record1 = {
    'gender': 'Female',
    'age': 22,
    'symptoms': '',
    'systemicManifestations': False,
    'ANA': 'Negative',
    'Anti-dsDNA': '',
    'RF': 'Low',
    'CRP': 'Normal',
    'WBC': 'High',
    'RBC': 'High',
    'Hemoglobin': 'Normal',
    'Platelets': '',
    'ESR': 'Low',
    'FVC': 'Normal',
    'FEV1': 'Normal',
    'FEV1/FVC Ratio': '',
    'Creatinine': '',
    'GFR': 'Normal',
    'C-Peptide': 'High',
    'Autoantibodies': '',
    'Fasting Glucose': '',
    'HbA1c': '',
    'Anti-CCP': '',
    'Blood Type': '',
    'Blood Pressure': '',
    'Heart Rate': '',
    'Respiratory Rate': '',
    'Body Temperature': '',
    'Oxygen Saturation': '',
    'Cholesterol': '',
    'ALT': '',
    'AST': '',
    'Current Medications': '',
    'X-ray Findings': '',
    'MRI Findings': '',
    'Echocardiogram Results': ''
}



# Prepare the test data
test_data = pd.DataFrame([Test_Medical_Record1])

# Encode the test data and account for data that may not match the training data
for column in test_data.columns:
    if column in label_encoders:
        le = label_encoders[column]
        test_data[column] = test_data[column].apply(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1)
        
        
    


# Predict the diagnosis
predicted_class_index = rf_Model.predict(test_data)[0]
predicted_proba = rf_Model.predict_proba(test_data)[0]

# Get the certainty percentage of the prediction
certainty_percentage = max(predicted_proba) * 100

# Convert the predicted label back to its original value
predicted_diagnosis = le_y.inverse_transform([predicted_class_index])[0]

# Display the prediction and certainty
print(f"Predicted Diagnosis: {predicted_diagnosis}, Certainty: {certainty_percentage:.2f}%")
=======
=======
>>>>>>> 9b98d0e (fixing file name)
from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}

<<<<<<< HEAD
# uvicorn main:app --reload
>>>>>>> d9f4cc1 (Structuring for api)
=======
# uvicorn main:app --reload
>>>>>>> 9b98d0e (fixing file name)
