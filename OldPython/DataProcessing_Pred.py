import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

data = pd.read_csv('Assignment3-Healthcare-Dataset.csv')

data = data.drop('LOSgroupNum', axis=1)

# Separate the dataset into numerical and categorical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Extract numerical data
numerical_data = data[numerical_cols]

# Apply MICE imputation on the numerical data
imputer = IterativeImputer(max_iter=10, random_state=0)
imputed_data = imputer.fit_transform(numerical_data)

# Convert the imputed data back to a DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=numerical_cols)

label_encoders = {}
# Impute missing values in categorical columns using the most frequent value
for col in categorical_cols:
    mode_val = data[col].mode()[0]
    data[col].fillna(mode_val, inplace=True)
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Update the dataset with the imputed numerical values
data[numerical_cols] = imputed_df

# Check if there are any missing values left in the dataset
remaining_missing = data.isnull().sum().sum()

print(remaining_missing)

data.to_csv('Dataset-Processed_Pred.csv', index=False)

new_data = pd.read_csv('Assignment3-Unknown-Dataset.csv')
new_data = new_data.drop('LOSgroupNum', axis=1)

# Separate the dataset into numerical and categorical columns
numerical_cols = new_data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = new_data.select_dtypes(include=['object']).columns

# Extract numerical data
numerical_data = new_data[numerical_cols]

# Apply MICE imputation on the numerical data
imputer = IterativeImputer(max_iter=10, random_state=0)
imputed_data = imputer.fit_transform(numerical_data)

# Convert the imputed data back to a DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=numerical_cols)

# Impute missing values in categorical columns using the most frequent value
for column, encoder in label_encoders.items():
    if column in new_data.columns:
        mode_val = data[column].mode()[0]
        data[column].fillna(mode_val, inplace=True)
        unseen_mask = ~new_data[column].isin(encoder.classes_)
        new_data.loc[unseen_mask, column] = encoder.classes_[0]
        new_data[column] = encoder.transform(new_data[column])

# Update the dataset with the imputed numerical values
new_data[numerical_cols] = imputed_df

new_data.to_csv('Dataset-Processed-Unknown_Pred.csv', index=False)

remaining_missing = new_data.isnull().sum().sum()

print(remaining_missing)