import requests

url = 'http://127.0.0.1:8000/predict/'
data = {
  "gender": "Female",
  "age": 22,
  "symptoms": "",
  "systemicManifestations": False,
  "ANA": "Negative",
  "Anti_dsDNA": "",
  "RF": "Low",
  "CRP": "Normal",
  "WBC": "High",
  "RBC": "High",
  "Hemoglobin": "Normal",
  "Platelets": "",
  "ESR": "Low",
  "FVC": "Normal",
  "FEV1": "Normal",
  "FEV1_FVC_Ratio": "",
  "Creatinine": "",
  "GFR": "Normal",
  "C_Peptide": "High",
  "Autoantibodies": "",
  "Fasting_Glucose": "",
  "HbA1c": "",
  "Anti_CCP": "",
  "Blood_Type": "",
  "Blood_Pressure": "",
  "Heart_Rate": "",
  "Respiratory_Rate": "",
  "Body_Temperature": "",
  "Oxygen_Saturation": "",
  "Cholesterol": "",
  "ALT": "",
  "AST": "",
  "Current_Medications": "",
  "X_ray_Findings": "",
  "MRI_Findings": "",
  "Echocardiogram_Results": ""
}




response = requests.post(url, json=data)

# Check the status code of the response
print("Status Code:", response.status_code)

# Print the raw response text
print("Response Text:", response.text)

# Only attempt to decode the response if it's successful
if response.status_code == 200:
    try:
        response_data = response.json()
        print(response_data)
    except ValueError:
        print("Response is not in JSON format.")
else:
    print("Request failed.")