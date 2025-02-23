import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.impute import SimpleImputer
import sys

print(sys.executable)

# Load dataset
data = pd.read_csv('/Users/macbookpro/Documents/Backup/OneDriveBack/Python Script/HeartAttack/heart_attack_streamlit_app/heart_attack_dataset.csv')

# Select features and target
features = ['Age', 'Gender', 'Cholesterol', 'BloodPressure', 'HeartRate', 'Smoker', 'Diabetes', 'Hypertension', 'FamilyHistory', 'StressLevel']
target = 'Outcome'

# Convert categorical features to numerical
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0}).fillna(0)
data['Smoker'] = data['Smoker'].map({'Yes': 1, 'No': 0}).fillna(0)
data['Diabetes'] = data['Diabetes'].map({'Yes': 1, 'No': 0}).fillna(0)
data['Hypertension'] = data['Hypertension'].map({'Yes': 1, 'No': 0}).fillna(0)
data['FamilyHistory'] = data['FamilyHistory'].map({'Yes': 1, 'No': 0}).fillna(0)

X = data[features]
y = data[target]

# Handle missing values
imputer = SimpleImputer(strategy='mean')  # You can use 'mean', 'median', or 'most_frequent'
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Save the model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')