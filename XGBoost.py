import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

data = pd.read_csv('Dataset-Training_SMOTE_Filled.csv')

# Split data into features (X) and target (y)
X = data.drop('ExpiredHospital', axis=1)
y = data['ExpiredHospital']

# Split data into training and testing sets (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train the model with the best parameters
clf = xgb.XGBClassifier()
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f"Accuracy on the test set: {accuracy:.4f}")
print(f"Precision on the test set: {precision:.4f}")
print(f"F1 Score on the test set: {f1:.4f}")

# Load unknown dataset
new_data = pd.read_csv('Dataset-Unknown_Base_Filled.csv')

predictions = clf.predict(new_data)

output = pd.DataFrame({
    'row ID': ["Row" + str(idx) for idx in new_data.index],
    'Predicted-ExpiredHospital': predictions
})

output.to_csv('predictions_XGBoost_SMOTE.csv', index=False)
