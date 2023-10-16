from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
from custom_encoder import CustomLabelEncoder


loaded_model = load_model('NN_Best.h5')
data = pd.read_csv('Assignment3-Unknown-Dataset.csv')

with open('custom_label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Update the label encoding in the preprocessing function using the CustomLabelEncoder
def preprocess_data(data, custom_label_encoders, scaler):
    # Handle missing values
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:
            data[column].fillna(data[column].median(), inplace=True)

    # Apply label encoding using the custom encoders from the training data
    for column, encoder in custom_label_encoders.items():
        if column in data.columns:
            data[column] = encoder.transform(data[column])

    # Scale the data using the scaler from the training data
    data_scaled = scaler.transform(data)

    return data_scaled


new_data_scaled = preprocess_data(data, label_encoders, scaler)

predictions = loaded_model.predict(new_data_scaled)
binary_predictions = (predictions > 0.5).astype(int)

# Create a dataframe with RowID and Predicted-ExpiredHospital
output = pd.DataFrame({
    'row ID': ["Row" + str(idx) for idx in range(len(new_data_scaled))],
    'Predicted-ExpiredHospital': binary_predictions.flatten()
})

# Save the output to a CSV (optional)
output.to_csv('predictions_NN.csv', index=False)
