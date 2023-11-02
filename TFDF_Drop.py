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

train_data, valid_data = train_test_split(train_data, test_size=0.20, random_state=42)

train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(train_data, task=tfdf.keras.Task.CLASSIFICATION,
                                                      label="ExpiredHospital")
valid_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(valid_data, task=tfdf.keras.Task.CLASSIFICATION,
                                                      label="ExpiredHospital")
predicting_data = tfdf.keras.pd_dataframe_to_tf_dataset(pred_data, task=tfdf.keras.Task.CLASSIFICATION)


model = tfdf.keras.GradientBoostedTreesModel(hyperparameter_template='benchmark_rank1', verbose=1)

model.fit(train_dataset)
model.compile(metrics=["accuracy", f1_score])
evaluation = model.evaluate(valid_dataset, return_dict=True)

for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")

X_train = train_data.drop(columns=["ExpiredHospital"])
y_train = train_data["ExpiredHospital"]

X_valid = valid_data.drop(columns=["ExpiredHospital"])
y_valid = valid_data["ExpiredHospital"]

# Assuming 'model' is the trained TFDF model
importances = model.make_inspector().variable_importances()["SUM_SCORE"]

# Convert to DataFrame for easier handling
importance_df = pd.DataFrame(importances, columns=["Feature", "Importance"])

# Set threshold - for example, based on percentile
threshold = importance_df["Importance"].quantile(0.05)

# Filter out features below threshold
selected_features = importance_df[importance_df["Importance"] > threshold]["Feature"].tolist()

for feature in selected_features:
    print(feature)

feature_names = [t[0] for t in selected_features]

# Use 'feature_names' to subset the data
reduced_train_data = pd.concat([X_train[feature_names], y_train], axis=1)
reduced_valid_data = pd.concat([X_valid[feature_names], y_valid], axis=1)

# Convert dataframes to tf datasets
train_dataset_dropped = tfdf.keras.pd_dataframe_to_tf_dataset(reduced_train_data, task=tfdf.keras.Task.CLASSIFICATION, label="ExpiredHospital")
valid_dataset_dropped = tfdf.keras.pd_dataframe_to_tf_dataset(reduced_valid_data, task=tfdf.keras.Task.CLASSIFICATION, label="ExpiredHospital")

# Train and evaluate model
model = tfdf.keras.GradientBoostedTreesModel(hyperparameter_template='benchmark_rank1', verbose=0)
model.fit(train_dataset_dropped)
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

output.to_csv('predictions_TFDF_drop.csv', index=False)
