
# test the api

import requests
import json

# test the api

medical_record = {
    'gender': 'Female',
    'age': 78,
    'blood_type': 'O+',
    'symptoms': 'fatigue|skin lesions|dry cough',
    'systemicManifestations': False,
    'finalDiagnosis': 'Systemic Lupus Erythematosus (SLE)',
    'ANA': 'Negative',
    'Anti-dsDNA': 'Positive',
    'RF': 'Not Tested',
    'CRP': 'High',
    'WBC': 'Low',
    'RBC': 'Low',
    'Hemoglobin': 'Low',
    'Platelets': 'Low',
    'ESR': 'High',
    'FVC': 'Normal',
    'FEV1': 'Reduced',
    'FEV1/FVC Ratio': 'Reduced',
    'Creatinine': 'Elevated',
    'GFR': 'Normal',
    'C-Peptide': 'Normal',
    'Autoantibodies': 'Negative',
    'Fasting Glucose': 'High',
    'Anti_dsDNA': 'Negative',
    'FEV1_FVC_Ratio': 'Reduced',
    'C_Peptide': 'Normal',
    'Fasting_Glucose': 'High',
    'FEV_FVC_RATIO': 'Reduced',
    'vital_signs_blood_pressure': 'Normal',
    'vital_signs_heart_rate': 78,
    'additional_blood_tests_lipid_profile_HDL': 'Low',
    'imaging_and_diagnostic_tests': 'Positive',
    'medication_and_treatment_history': 'Positive',
    'HbA1c': 'High',
    'Anti_CCP': 'Negative'
    
    
}


# send a post request to the prediction api
response = requests.post('http://localhost:8000/predict', json=medical_record)

# print the response
print(response.json())

    