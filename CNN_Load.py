from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Dataset-Processed-Unknown_Filled.csv")
model = load_model('CNN.h5')

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

predictions = model.predict(data_scaled)
binary_predictions = (predictions > 0.5).astype(int)

output = pd.DataFrame({
    'row ID': ["Row" + str(idx) for idx in data.index],
    'Predicted-ExpiredHospital': binary_predictions.flatten()
})

output.to_csv('predictions_CNN.csv', index=False)
