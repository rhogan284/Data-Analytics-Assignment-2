import pandas as pd
from sklearn.preprocessing import LabelEncoder

def fill_missing_values(df, categorical_columns):
    for column in df.columns:
        if column in categorical_columns:
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].median(), inplace=True)


def encode_categorical(df, encoders=None):
    if encoders is None:
        encoders = {}
        is_training = True
    else:
        is_training = False

    for column in df.select_dtypes(include=['object']).columns:
        if is_training:
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column])
            encoders[column] = encoder
        else:
            unseen_mask = ~df[column].isin(encoders[column].classes_)
            df.loc[unseen_mask, column] = encoders[column].classes_[0]
            df[column] = encoders[column].transform(df[column])

    return encoders

def process_dataset(file_path, encoders=None):
    df = pd.read_csv(file_path)
    df = df.drop('LOSgroupNum', axis=1)
    categorical_columns = df.select_dtypes(include=['object']).columns
    fill_missing_values(df, categorical_columns)
    encoders = encode_categorical(df, encoders)
    return df, encoders

# Process training data
data, label_encoders = process_dataset('Assignment3-Healthcare-Dataset.csv')
data.to_csv('Dataset-Processed_Filled_3.csv', index=False)

# Process new data
new_data, _ = process_dataset('Assignment3-Unknown-Dataset.csv', label_encoders)
new_data.to_csv('Dataset-Processed-Unknown_Filled_3.csv', index=False)
