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


# Apply feature engineering to both training and prediction datasets
train_data = pd.read_csv('Dataset-Processed_Filled.csv')
pred_data = pd.read_csv('Dataset-Processed-Unknown_Filled.csv')

train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=42)

train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(train_data, task=tfdf.keras.Task.CLASSIFICATION, label="ExpiredHospital")
valid_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(valid_data, task=tfdf.keras.Task.CLASSIFICATION, label="ExpiredHospital")
predicting_data = tfdf.keras.pd_dataframe_to_tf_dataset(pred_data, task=tfdf.keras.Task.CLASSIFICATION)

model = tfdf.keras.GradientBoostedTreesModel(
    hyperparameters={
        "shrinkage": 0.05,
        "min_examples": 10,
        "max_num_nodes": 16,
        "growing_strategy": "BEST_FIRST_GLOBAL",
        "sampling_method": "RANDOM",
        "subsample": 1.0,
        "categorical_algorithm": "CART",
        "num_candidate_attributes_ratio": 1.0,
        "use_hessian_gain": True,
    },
    task=tfdf.keras.Task.CLASSIFICATION
)

model.fit(train_dataset, verbose=2)
model.compile(metrics=["accuracy", f1_score])
evaluation = model.evaluate(valid_dataset, return_dict=True)

for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")

tuning_logs = model.make_inspector().tuning_logs()
print(tuning_logs[tuning_logs.best].iloc[0])

plt.figure(figsize=(10, 5))
plt.plot(tuning_logs["score"], label="current trial")
plt.plot(tuning_logs["score"].cummax(), label="best trial")
plt.xlabel("Tuning step")
plt.ylabel("Tuning score")
plt.legend()
plt.show()