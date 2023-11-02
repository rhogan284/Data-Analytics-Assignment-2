import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

data = pd.read_csv('Dataset-Processed_Filled_2.csv')

# Split data into features (X) and target (y)
X = data.drop('ExpiredHospital', axis=1)
y = data['ExpiredHospital']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

# 1. Create an XGBoost classifier instance
clf = xgb.XGBClassifier()

# 2. Define the hyperparameter grid to search
param_grid = {
    'colsample_bytree': [0.5, 0.7, 0.9],
    'gamma': [0.0, 0.1, 0.2, 0.3],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5, 6, 7],
    'n_estimators': [100, 150, 200, 250],
    'subsample': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
}

f1_scorer = make_scorer(f1_score)

# 3. Set up the grid search
grid_clf = GridSearchCV(clf, param_grid, cv=3, scoring=f1_scorer, n_jobs=-1, verbose=2)

# 4. Train the models with different hyperparameters
grid_clf.fit(X_train_scaled, y_resampled)

# 5. Get the best parameters and best model
best_params = grid_clf.best_params_
best_model = grid_clf.best_estimator_

print(f"Best Parameters: {best_params}")

# Optional: Evaluate the best model on the test set
y_pred = best_model.predict(X_test_scaled)
precision = precision_score(y_resampled, y_pred)
print(f"Precision of the best model on the test set: {precision:.4f}")

