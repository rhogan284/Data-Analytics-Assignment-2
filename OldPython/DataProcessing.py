import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

def fill_missing_values(df, numeric_columns):
    for column in df.columns:
        if column in numeric_columns:
            df[column].fillna(df[column].median(), inplace=True)
        else:
            df[column].fillna(df[column].mode()[0], inplace=True)


def fill_missing_values_with_mice(df, encoders=None):
    categorical_columns = df.select_dtypes(include=['object']).columns
    if encoders is None:
        encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_columns}
        df = df.apply(lambda col: encoders[col.name].transform(col) if col.name in categorical_columns else col)

    mice_imputer = IterativeImputer()
    df_imputed = pd.DataFrame(mice_imputer.fit_transform(df), columns=df.columns)

    for column in categorical_columns:
        df_imputed[column] = encoders[column].inverse_transform(df_imputed[column].astype(int))

    return df_imputed, encoders

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

def normalize_features(df, scaler=None, is_training=True):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if scaler is None:
        scaler = MinMaxScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    else:
        if is_training:
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        else:
            df[numeric_columns] = scaler.transform(df[numeric_columns])
    return scaler

def apply_smote_to_df(df, target_column):
    smote = SMOTE()
    X, y = df.drop(target_column, axis=1), df[target_column]
    X_res, y_res = smote.fit_resample(X, y)
    return pd.DataFrame(X_res, columns=X.columns).assign(**{target_column: y_res})

def process_dataset(file_path, encoders=None, scaler=None, use_mice=False, apply_smote=False, target_column=None):
    df = pd.read_csv(file_path).drop('LOSgroupNum', axis=1)

    if use_mice:
        df, encoders = fill_missing_values_with_mice(df, encoders)
    else:
        fill_missing_values(df, df.select_dtypes(include=['number']).columns)

    if apply_smote and target_column:
        df = apply_smote_to_df(df, target_column)

    scaler = normalize_features(df, scaler)
    encoders = encode_categorical(df, encoders)
    return df, encoders


# Process training data with SMOTE
data, label_encoders = process_dataset('Assignment3-Healthcare-Dataset.csv', use_mice=False, apply_smote=False,
                                               target_column='ExpiredHospital')
data.to_csv('Dataset-Processed_Base.csv', index=False)

print("Train: Base Complete")

data, label_encoders = process_dataset('Assignment3-Healthcare-Dataset.csv', use_mice=False, apply_smote=True,
                                               target_column='ExpiredHospital')
data.to_csv('Dataset-Processed_SMOTE.csv', index=False)

print("Train: SMOTE Complete")

data, label_encoders = process_dataset('Assignment3-Healthcare-Dataset.csv', use_mice=True, apply_smote=False,
                                               target_column='ExpiredHospital')
data.to_csv('Dataset-Processed_MICE.csv', index=False)

print("Train: MICE Complete")

data, label_encoders = process_dataset('Assignment3-Healthcare-Dataset.csv', use_mice=True, apply_smote=True,
                                               target_column='ExpiredHospital')
data.to_csv('Dataset-Processed_SMOTE+MICE.csv', index=False)

print("Train: SMOTE+MICE Complete")

# Process new data without SMOTE
new_data, _ = process_dataset('Assignment3-Unknown-Dataset.csv', label_encoders, use_mice=False)
new_data.to_csv('Dataset-Processed-Unknown_Base.csv', index=False)

print("Unknown: Base Complete")

new_data, _ = process_dataset('Assignment3-Unknown-Dataset.csv', label_encoders, use_mice=True)
new_data.to_csv('Dataset-Processed-Unknown_MICE.csv', index=False)

print("Unknown: MICE Complete")