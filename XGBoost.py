import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

data = pd.read_csv('Dataset-Processed_Filled.csv')

# Split data into features (X) and target (y)
X = data.drop('ExpiredHospital', axis=1)
y = data['ExpiredHospital']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

smote = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

# Given the best parameters:
best_params = {
    'colsample_bytree': 0.9,
    'gamma': 0.2,
    'learning_rate': 0.3,
    'max_depth': 5,
    'n_estimators': 150,
    'subsample': 1
}

clf = xgb.XGBClassifier(**best_params)

clf.fit(X_train_scaled, y_resampled)

predicitons = clf.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predicitons)
print(f"Accuracy on the test set: {accuracy:.4f}")
accuracy = precision_score(y_test, predicitons)
print(f"Precision on the test set: {accuracy:.4f}")

# # Load unknown dataset
# new_data = pd.read_csv('Dataset-Processed-Unknown_Pred.csv')
#
# new_data_scaled = scaler.transform(new_data)
#
# predictions = clf.predict(new_data_scaled)
#
# output = pd.DataFrame({
#     'row ID': ["Row" + str(idx) for idx in new_data.index],
#     'Predicted-ExpiredHospital': predictions
# })
#
# output.to_csv('predictions_XGBoost_Pred_v3.csv', index=False)
