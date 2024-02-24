import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



# Load JSON data
with open('Test_Medical_Records2.Test_Medical_Records2.json') as file:
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
        label_encoders[column] = le

        


# Split data into features and target variable
X = df.drop('Final Diagnosis', axis=1)  # Replace 'finalDiagnosis' with your actual target column name if different
y = df['Final Diagnosis']  # Replace 'finalDiagnosis' with your actual target column name if different

# Optionally, encode the target variable if it's categorical
le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)

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

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

rf_Model = RandomForestClassifier(n_estimators=100, random_state=23123)
rf_Model.fit(X_train, y_train)
rf_Model.score(X_test, y_test)
y_pred = rf_Model.predict(X_test)

# print out the accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
# print out the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print out the confusion matrix
from sklearn.metrics import classification_report
print("Classification Report:", classification_report(y_test, y_pred))

# compare model predictions with actual values
print("Predictions:", y_pred)
print("Actual:     ", y_test)

# print out the accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
