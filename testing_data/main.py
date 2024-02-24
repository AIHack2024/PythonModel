import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

data1 = pd.read_csv("Test_Medical_Records.Test_Medical_Records.csv")


# Handle missing values
# Assuming 'data' is your DataFrame
imputer = SimpleImputer(strategy='most_frequent')
data_imputed = pd.DataFrame(imputer.fit_transform(data1), columns=data1.columns)


# connect to Test_Medical_Records.Test_Medical_Records.csv
# Read the dataset

encoder = OneHotEncoder()

categorical_cols = ['personalInformation.gender', 'clinicalPresentation.symptoms[0]', 'finalDiagnosis']  # Example columnsencoder = OneHotEncoder()
data_encoded = encoder.fit_transform(data_imputed[categorical_cols])
data_encoded_df = pd.DataFrame(data_encoded.toarray(), columns=encoder.get_feature_names_out())
# Drop original categorical columns and concatenate encoded ones
data_imputed.drop(columns=categorical_cols, inplace=True)
data_final = pd.concat([data_imputed, data_encoded_df], axis=1)

# Drop the 'id' column
data = data_final.drop('_id', axis=1)

print(data.head())
print(data.info())
print(data.describe())
print("null:", data.isnull().sum())

Y = data['finalDiagnosis']
X = data.drop('finalDiagnosis', axis=1)

print("X Head:", X.head())
print("Y Head", Y.head())

print("X Shape(Rows, Variables):", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("X Train Shape:", X_train.shape)
print("X Test Shape:", X_test.shape)
print("Y Train Shape:", y_train.shape)
print("Y Test Shape:", y_test.shape)


rf_Model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_Model.fit(X_train, y_train)
rf_Model.score(X_test, y_test)
y_pred = rf_Model.predict(X_test)

# print out the accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
# print out the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
# print out the classification report
from sklearn.metrics import classification_report   
print("Classification Report:", classification_report(y_test, y_pred))

