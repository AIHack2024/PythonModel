import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

encoder = OneHotEncoder()

param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=64), param_grid, cv=5, verbose=2, n_jobs=-1)


# Load JSON data
with open('csvjson5K.json') as file:
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


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Optionally, encode the target variable if it's categorical
le_y = LabelEncoder()
# print the values of the target variable
y_encoded = le_y.fit_transform(y)
# print("Target Variable Values:", le_y.classes_)




X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

grid_search.fit(X_train, y_train)

# Print the best parameters and use them to create the best model
print(f"Best parameters found: {grid_search.best_params_}")
best_model = grid_search.best_estimator_



# Get feature importances from the model
feature_importances = best_model.feature_importances_

# Sort the feature importances in descending order and plot
sorted_idx = np.argsort(feature_importances)[::-1]

# plt.figure(figsize=(10, 7))
# plt.title("Feature Importances")
# plt.bar(range(X.shape[1]), feature_importances[sorted_idx], align="center")
# plt.xticks(range(X.shape[1]), X.columns[sorted_idx], rotation=90)
# plt.xlim([-1, X.shape[1]])
# plt.show()

rf_Model = RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100 , random_state=64)
rf_Model.fit(X_train, y_train)
rf_Model.score(X_test, y_test)
y_pred = rf_Model.predict(X_test)

cv_scores = cross_val_score(best_model, X, y_encoded, cv=5)
print(f"CV Average Score: {cv_scores.mean()}")



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
    'gender': 'Male',
    'age': 52,
    'blood_type': 'A-',
    'symptoms': 'abdominal pain|weight loss|bloody diarrhea',
    'systemicManifestations': False,
    'finalDiagnosis': "Crohn's Disease",
    'ANA': 'Negative',
    'Anti-dsDNA': 'Positive',
    'RF': 'Not Tested',
    'CRP': 'Normal',
    'WBC': 'Elevated',
    'RBC': 'Low',
    'Hemoglobin': 'Low',
    'Platelets': 'Normal',
    'ESR': 'Normal',
    'FVC': 'Reduced',
    'FEV1': 'Reduced',
    'FEV1/FVC Ratio': 'Reduced',
    'Creatinine': 'Elevated',
    'GFR': 'Reduced',
    'C-Peptide': 'Low',
    'Autoantibodies': 'Positive',
    'Fasting Glucose': 'Not Tested',
    'HbA1c': 'Normal',
    'Anti-CCP': 'Not Tested',
    'vital_signs_blood_pressure': '91/70',
    'vital_signs_heart_rate': '',
    'additional_blood_tests_lipid_profile_HDL': '',
    'imaging_and_diagnostic_tests': 'CT scan normal',
    'medication_and_treatment_history': ''
}





# Prepare the test data
test_data = pd.DataFrame([Test_Medical_Record1])



# Encode the test data and account for data that may not match the training data
for column in test_data.columns:
    if column in label_encoders:
        le = label_encoders[column]
        test_data[column] = test_data[column].apply(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1)
        
        
    
# if finalDiagnosis is in the test data, remove it
if 'finalDiagnosis' in test_data.columns:
    test_data = test_data.drop('finalDiagnosis', axis=1)


# Predict the diagnosis
predicted_class_index = rf_Model.predict(test_data)[0]
predicted_proba = rf_Model.predict_proba(test_data)[0]

# Get the certainty percentage of the prediction
certainty_percentage = max(predicted_proba) * 100

# Convert the predicted label back to its original value
predicted_diagnosis = le_y.inverse_transform([predicted_class_index])[0]

# Display the prediction and certainty
print(f"Predicted Diagnosis: {predicted_diagnosis}, Certainty: {certainty_percentage:.2f}%")


