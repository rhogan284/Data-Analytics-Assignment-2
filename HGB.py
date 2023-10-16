from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Dataset-Processed_Filled.csv')

# Split data into features (X) and target (y)
X = data.drop('ExpiredHospital', axis=1)
y = data['ExpiredHospital']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the base models
clf = HistGradientBoostingClassifier().fit(X_train_scaled, y_train)
reg = HistGradientBoostingRegressor().fit(X_train_scaled, y_train)

# Get predictions for the scaled test set
clf_probs = clf.predict_proba(X_test_scaled)[:, 1]  # Probabilities for the positive class
reg_preds = reg.predict(X_test_scaled)

# Stack predictions to form new features for the validation set
stacked_features_val = np.column_stack([clf_probs, reg_preds])

# Train the meta-model
meta_model = HistGradientBoostingClassifier().fit(stacked_features_val, y_test)

predictions = meta_model.predict(stacked_features_val)

HGB_accuracy = accuracy_score(y_test, predictions)
print(HGB_accuracy)

# Load unknown dataset
new_data = pd.read_csv('Dataset-Processed-Unknown_Filled.csv')

new_data_scaled = scaler.transform(new_data)

# Obtain predictions from base models for the new_data_scaled
clf_probs_new = clf.predict_proba(new_data_scaled)[:, 1]
reg_preds_new = reg.predict(new_data_scaled)

# Stack predictions to form new features
stacked_features_new = np.column_stack([clf_probs_new, reg_preds_new])

# Predict with meta-model using the stacked features
predictions = meta_model.predict(stacked_features_new)

output = pd.DataFrame({
    'row ID': ["Row" + str(idx) for idx in new_data.index],
    'Predicted-ExpiredHospital': predictions
})

output.to_csv('predictions_HGB_filled.csv', index=False)
