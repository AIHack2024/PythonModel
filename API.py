from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib  # For saving and loading models

app = Flask(__name__)

# Initialize your model and other necessary components here
# For example, you might load a saved model and label encoders
try:
    rf_model = joblib.load("model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    le_y = joblib.load("label_encoders_y.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print(e)
    rf_model = None
    label_encoders = None
    le_y = None
    scaler = None

@app.route('/')
def home():
    return "Machine Learning Model API"

@app.route('/predict', methods=['POST'])
def predict():
    if not rf_model:
        return jsonify({'error': 'Model is not loaded.'}), 500

    data = request.json
    test_data = pd.DataFrame([data])

    # Preprocess the input data in the same way as during training
    for column in test_data.columns:
        if column in label_encoders and column in test_data:
            le = label_encoders[column]
            test_data[column] = test_data[column].apply(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1)

    if 'finalDiagnosis' in test_data.columns:
        test_data = test_data.drop('finalDiagnosis', axis=1)
    
    # Scale the data
    test_data_scaled = scaler.transform(test_data)

    # Make a prediction
    predicted_class_index = rf_model.predict(test_data_scaled)[0]
    predicted_proba = rf_model.predict_proba(test_data_scaled)[0]
    certainty_percentage = max(predicted_proba) * 100
    predicted_diagnosis = le_y.inverse_transform([predicted_class_index])[0]

    return jsonify({'Predicted Diagnosis': predicted_diagnosis, 'Certainty': f"{certainty_percentage:.2f}%"})

if __name__ == '__main__':
    app.run(debug=True)
