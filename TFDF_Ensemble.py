import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow_decision_forests as tfdf
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score as sk_f1_score
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
import xgboost as xgb

from imblearn.over_sampling import SMOTE, ADASYN


def f1_score(y_true, y_pred):
    """Compute the F1 score."""
    y_pred_binary = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred_binary, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred_binary), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred_binary, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred_binary), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2 * p * r / (p + r + K.epsilon())
    return K.mean(tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1))


def add_engineered_features(data):
    """Add engineered features to the dataset."""
    interaction_features = [
        ('NumLabs', 'TotalNumInteract'),
        ('NumLabs', 'NumChartEvents'),
        ('TotalNumInteract', 'NumChartEvents')
    ]
    for f1, f2 in interaction_features:
        data[f'{f1}_{f2}'] = data[f1] * data[f2]

    log_transform_features = ['NumLabs', 'TotalNumInteract', 'NumChartEvents', 'NumInput', 'NumCPTevents']
    for feature in log_transform_features:
        data[f'log_{feature}'] = np.log1p(data[feature])

    return data


def apply_adasyn(data):
    """Apply ADASYN oversampling to the dataset."""
    X = data.drop(columns=["ExpiredHospital"])
    y = data["ExpiredHospital"]
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    return pd.concat([X_resampled, y_resampled], axis=1)


def apply_smote(data):
    """Apply SMOTE oversampling to the dataset."""
    X = data.drop(columns=["ExpiredHospital"])
    y = data["ExpiredHospital"]
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return pd.concat([X_resampled, y_resampled], axis=1)


# Load datasets
train_data = pd.read_csv('Dataset-Processed_Base.csv')
train_dataE = pd.read_csv('Assignment3-Healthcare-Dataset.csv')
pred_data = pd.read_csv('Dataset-Processed-Unknown_Base.csv')

# Split training data
train_data, valid_data = train_test_split(train_data, test_size=0.20, random_state=42)

# # Apply SMOTE to training data
# train_data = apply_smote(train_data)

# Convert to TF datasets
train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(train_data, task=tfdf.keras.Task.CLASSIFICATION, label="ExpiredHospital")
train_datasetE = tfdf.keras.pd_dataframe_to_tf_dataset(train_dataE, task=tfdf.keras.Task.CLASSIFICATION, label="ExpiredHospital")
valid_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(valid_data, task=tfdf.keras.Task.CLASSIFICATION, label="ExpiredHospital")
predicting_data = tfdf.keras.pd_dataframe_to_tf_dataset(pred_data, task=tfdf.keras.Task.CLASSIFICATION)

# Model training
models = []

tfdf_model = tfdf.keras.GradientBoostedTreesModel(hyperparameter_template="benchmark_rank1", verbose=1)
tfdf_model.fit(train_dataset)
models.append(tfdf_model)

tfdf_model2 = tfdf.keras.GradientBoostedTreesModel(verbose=1)
tfdf_model2.fit(train_dataset)
models.append(tfdf_model2)

best_params = {
    'colsample_bytree': 0.9,
    'gamma': 0.2,
    'learning_rate': 0.3,
    'max_depth': 5,
    'n_estimators': 150,
    'subsample': 1
}
xgb_model = xgb.XGBClassifier(**best_params)
xgb_model.fit(train_data.drop('ExpiredHospital', axis=1), train_data['ExpiredHospital'])
models.append(xgb_model)

# Model predictions
base_model_predictions = []

for model in models:
    if isinstance(model, xgb.XGBClassifier):
        preds = model.predict_proba(pred_data)[:, 1]
    elif isinstance(model, tfdf.keras.GradientBoostedTreesModel):
        preds = model.predict(predicting_data).squeeze()
    else:
        preds = model.predict(pred_data)

    base_model_predictions.append(preds)

# Ensemble predictions
avg_predictions = np.mean(base_model_predictions, axis=0)
binary_predictions = (avg_predictions > 0.5).astype(int)

output = pd.DataFrame({
    'row ID': ["Row" + str(idx) for idx in pred_data.index],
    'Predicted-ExpiredHospital': binary_predictions
})
output.to_csv('predictions_TFDF_ensemble_Test3.csv', index=False)
