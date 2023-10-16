import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

data = pd.read_csv('Assignment3-Healthcare-Dataset.csv')

data = data.drop('LOSgroupNum', axis=1)

for column in data.columns:
    if data[column].dtype == 'object':
        data[column].fillna(data[column].mode()[0], inplace=True)
    else:
        data[column].fillna(data[column].median(), inplace=True)

label_encoders = {}
categorical_columns = data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

data.to_csv('Dataset-Processed_Filled.csv', index=False)

new_data = pd.read_csv('Assignment3-Unknown-Dataset.csv')

new_data = new_data.drop('LOSgroupNum', axis=1)

for column in new_data.columns:
    if column in categorical_columns:
        new_data[column].fillna(new_data[column].mode()[0], inplace=True)
    else:
        new_data[column].fillna(new_data[column].median(), inplace=True)


for column, encoder in label_encoders.items():
    if column in new_data.columns:
        # Get the mask for unseen labels
        unseen_mask = ~new_data[column].isin(encoder.classes_)

        # Temporarily replace unseen labels with a known label
        new_data.loc[unseen_mask, column] = encoder.classes_[0]

        # Now transform the data
        new_data[column] = encoder.transform(new_data[column])

new_data.to_csv('Dataset-Processed-Unknown_Filled.csv', index=False)