import tensorflow as tf
from tensorflow import keras
import tensorflow_decision_forests as tfdf
import numpy as np
import pandas as pd

data = pd.read_csv('Assignment3-Unknown-Dataset.csv')
predicting_data = data.drop(columns=["LOSgroupNum"])
predicting_data = tfdf.keras.pd_dataframe_to_tf_dataset(predicting_data, task=tfdf.keras.Task.CLASSIFICATION)

model = tf.keras.models.load_model('TFDF.keras')

predictions = model.predict(predicting_data, verbose=1)
binary_predictions = (predictions > 0.5).astype(int)

output = pd.DataFrame({
    'row ID': ["Row" + str(idx) for idx in data.index],
    'Predicted-ExpiredHospital': binary_predictions.flatten()
})

output.to_csv('predictions_TFDF.csv', index=False)