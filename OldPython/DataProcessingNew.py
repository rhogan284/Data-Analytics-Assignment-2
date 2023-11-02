import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
file_path = 'Assignment3-Healthcare-Dataset.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Remove the 'LOSgroupNum' column
data_cleaned = data.drop(columns=['LOSgroupNum'])

# Impute missing values
# For 'age', use the median
age_median = data_cleaned['age'].median()
data_cleaned['age'].fillna(age_median, inplace=True)

# For categorical columns ('marital_status', 'religion', 'AdmitDiagnosis'), use the mode (most frequent value)
for column in ['marital_status', 'religion', 'AdmitDiagnosis']:
    most_frequent = data_cleaned[column].mode()[0]
    data_cleaned[column].fillna(most_frequent, inplace=True)

# Impute missing values in 'NumCallouts' with the median
num_callouts_median = data_cleaned['NumCallouts'].median()
data_cleaned['NumCallouts'].fillna(num_callouts_median, inplace=True)

# Identifying categorical and numerical columns
categorical_cols = data_cleaned.select_dtypes(include=['object']).columns
numerical_cols = data_cleaned.select_dtypes(include=['int64', 'float64']).columns
numerical_cols = numerical_cols.drop('ExpiredHospital')  # Exclude the target variable

# Limiting the number of categories for each categorical variable
for col in categorical_cols:
    # Find the top 10 most frequent categories for each categorical column
    top_categories = data_cleaned[col].value_counts().index[:10]

    # Replace categories not in the top 10 with 'Other'
    data_cleaned.loc[~data_cleaned[col].isin(top_categories), col] = 'Other'

# Process numerical columns
scaler = StandardScaler()
data_numerical_scaled = scaler.fit_transform(data_cleaned[numerical_cols])

# Process categorical columns with sparse output
encoder_sparse = OneHotEncoder(sparse=True)
data_categorical_encoded_sparse = encoder_sparse.fit_transform(data_cleaned[categorical_cols])

# Convert the sparse matrix to a DataFrame
categorical_columns = encoder_sparse.get_feature_names(categorical_cols)
data_categorical_encoded_df = pd.DataFrame.sparse.from_spmatrix(data_categorical_encoded_sparse, columns=categorical_columns)

# Concatenate the numerical and new categorical data
data_final_preprocessed = pd.concat([data_cleaned[numerical_cols].reset_index(drop=True), data_categorical_encoded_df.reset_index(drop=True)], axis=1)

# Display the first few rows of the final preprocessed data
print(data_final_preprocessed.head())
