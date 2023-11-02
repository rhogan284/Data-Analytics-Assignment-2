import tensorflow as tf
from tensorflow import keras
import tensorflow_decision_forests as tfdf
import os
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from sklearn.metrics import f1_score as sk_f1_score
from imblearn.over_sampling import ADASYN


def f1_score(y_true, y_pred):
    """Compute the F1 score."""
    # Convert probabilities to binary predictions
    y_pred_binary = K.round(y_pred)

    tp = K.sum(K.cast(y_true * y_pred_binary, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred_binary), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred_binary, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred_binary), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def add_engineered_features(data):
    """Function to add engineered features to the dataset."""
    # Interaction features
    data['NumLabs_TotalNumInteract'] = data['NumLabs'] * data['TotalNumInteract']
    data['NumLabs_NumChartEvents'] = data['NumLabs'] * data['NumChartEvents']
    data['TotalNumInteract_NumChartEvents'] = data['TotalNumInteract'] * data['NumChartEvents']

    # Log transformation for top continuous features (adding 1 to avoid log(0))
    for feature in ['NumLabs', 'TotalNumInteract', 'NumChartEvents', 'NumInput', 'NumCPTevents']:
        data[f'log_{feature}'] = np.log1p(data[feature])

    return data


def apply_adasyn(data):
    """Apply ADASYN oversampling to the dataset."""

    # Separate features and target
    X = data.drop(columns=["ExpiredHospital"])
    y = data["ExpiredHospital"]

    # Apply ADASYN
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)

    # Combine resampled features and target into a DataFrame
    data_resampled = pd.concat([X_resampled, y_resampled], axis=1)

    return data_resampled


# Apply feature engineering to both training and prediction datasets
train_data = pd.read_csv('Dataset-Processed_Filled.csv')
train_data = apply_adasyn(train_data)
pred_data = pd.read_csv('Dataset-Processed-Unknown_Filled.csv')

train_data, valid_data = train_test_split(train_data, test_size=0.20, random_state=42)

train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(train_data, task=tfdf.keras.Task.CLASSIFICATION,
                                                      label="ExpiredHospital")
valid_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(valid_data, task=tfdf.keras.Task.CLASSIFICATION,
                                                      label="ExpiredHospital")
predicting_data = tfdf.keras.pd_dataframe_to_tf_dataset(pred_data, task=tfdf.keras.Task.CLASSIFICATION)

model = tfdf.keras.RandomForestModel(hyperparameter_template='benchmark_rank1', verbose=1)

model.fit(train_dataset)
model.compile(metrics=["accuracy", f1_score])
evaluation = model.evaluate(valid_dataset, return_dict=True)

for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")

predictions = model.predict(predicting_data, verbose=1)
binary_predictions = (predictions > 0.5).astype(int)

output = pd.DataFrame({
    'row ID': ["Row" + str(idx) for idx in pred_data.index],
    'Predicted-ExpiredHospital': binary_predictions.flatten()
})

output.to_csv('predictions_TFDF_RF.csv', index=False)
