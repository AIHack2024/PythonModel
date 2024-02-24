import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



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
print("Target Variable Values:", le_y.classes_)

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
from sklearn.metrics import confusion_matrix
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print out the confusion matrix
from sklearn.metrics import classification_report
# print("Classification Report:", classification_report(y_test, y_pred))

# # compare model predictions with actual values
# print("Predictions:", y_pred)
# print("Actual:     ", y_test)

# # print out the accuracy
# print("Accuracy:", accuracy_score(y_test, y_pred))


# test the model with specific data

Test_Medical_Record1 = {
    'gender': 'Male',
    'age': 61,
    'symptoms': 'eye inflammation|genital ulcers|mouth ulcers',
    'systemicManifestations': True,
    'ANA': 'Negative',
    'Anti-dsDNA': 'Positive',
    'RF': 'Positive',
    'CRP': 'Normal',
    'WBC': 'Elevated',
    'RBC': 'Normal',
    'Hemoglobin': 'Normal',
    'Platelets': 'Elevated',
    'ESR': 'Elevated',
    'FVC': 'Reduced',
    'FEV1': 'Reduced',
    'FEV1/FVC Ratio': 'Reduced',
    'Creatinine': 'Normal',
    'GFR': 'Normal',
    'C-Peptide': 'Low',
    'Autoantibodies': 'Negative',
    'Fasting Glucose': 'Not Tested',
    'HbA1c': 'Normal',
    'Anti-CCP': 'Not Tested',
    'Blood Type': 'A-',
    'Blood Pressure': '109/87',
    'Heart Rate': '65',
    'Respiratory Rate': '16',
    'Body Temperature': '',
    'Oxygen Saturation': '',
    'Cholesterol': '200',
    'ALT': '36',
    'AST': '28',
    'Current Medications': 'Medication A; Dosage: 1 daily',
    'X-ray Findings': '',
    'MRI Findings': '',
    'Echocardiogram Results': ''
}

# test the single data point Test_Medical_Record1
# Convert the data to a DataFrame
df_test = pd.DataFrame([Test_Medical_Record1])

# Normalize JSON data into a flat table
df_test = pd.json_normalize(Test_Medical_Record1)

# Automatically handle columns with lists
for column in df_test.columns:
    if df_test[column].apply(lambda x: isinstance(x, list)).any():
        df_test[column] = df_test[column].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else x)

# Encode categorical variables
label_encoders = {}
# Convert columns to strings before encoding
for column in df_test.select_dtypes(include=['object']).columns:
    if column != 'finalDiagnosis':  # Assuming 'finalDiagnosis' is the target variable
        # Convert the column to string type
        df_test[column] = df_test[column].astype(str)  # Ensure all data is string type

        le = LabelEncoder()
        df_test[column] = le.fit_transform(df_test[column])
        label_encoders[column] = le
        
        
        
# print out the prediction of the final diagnosis
print("Predicted Final Diagnosis:", le_y.inverse_transform(rf_Model.predict(df_test)))

# print out the total certainty of the prediction
print("Certainty of Prediction:", rf_Model.predict_proba(df_test))
