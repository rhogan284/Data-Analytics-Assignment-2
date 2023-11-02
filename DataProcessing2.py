import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def fill_missing_values(df, categorical_columns, use_mice=False):
    if not use_mice:
        for column in df.columns:
            if column in categorical_columns:
                df[column].fillna(df[column].mode()[0], inplace=True)
            else:
                df[column].fillna(df[column].median(), inplace=True)
    else:
        imputer = IterativeImputer(max_iter=10, random_state=0)
        df[:] = imputer.fit_transform(df)



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


def process_dataset(file_path, encoders=None, apply_smote=False, use_mice=False):
    df = pd.read_csv(file_path)
    df = df.drop('LOSgroupNum', axis=1)
    categorical_columns = df.select_dtypes(include=['object']).columns

    encoders = encode_categorical(df, encoders)
    fill_missing_values(df, categorical_columns, use_mice)

    if apply_smote:
        smote = SMOTE(random_state=42)
        X = df.drop('ExpiredHospital', axis=1)
        y = df['ExpiredHospital']
        X, y = smote.fit_resample(X, y)
        df = pd.concat([X, y], axis=1)

    return df, encoders


# Process datasets
# 1. Training Base (Filled)
train_base, label_encoders = process_dataset('Assignment3-Healthcare-Dataset.csv')
train_base.to_csv('Dataset-Training_Base_Filled.csv', index=False)
print("Training Base Complete")

# 2. Unknown Base (Filled)
unknown_base, _ = process_dataset('Assignment3-Unknown-Dataset.csv', label_encoders)
unknown_base.to_csv('Dataset-Unknown_Base_Filled.csv', index=False)
print("Unknown Base Complete")

# 3. Training SMOTE
train_smote, _ = process_dataset('Assignment3-Healthcare-Dataset.csv', label_encoders, apply_smote=True)
train_smote.to_csv('Dataset-Training_SMOTE_Filled.csv', index=False)
print("Training SMOTE Complete")

# 4. Training MICE
train_mice, label_encoders_mice = process_dataset('Assignment3-Healthcare-Dataset.csv', use_mice=True)
train_mice.to_csv('Dataset-Training_MICE_Filled.csv', index=False)
print("Training MICE Complete")

# 5. Unknown MICE
unknown_mice, _ = process_dataset('Assignment3-Unknown-Dataset.csv', label_encoders_mice, use_mice=True)
unknown_mice.to_csv('Dataset-Unknown_MICE_Filled.csv', index=False)
print("Unknown MICE Complete")

# 6. Training MICE + SMOTE
train_mice_smote, _ = process_dataset('Assignment3-Healthcare-Dataset.csv', label_encoders, apply_smote=True, use_mice=True)
train_mice_smote.to_csv('Dataset-Training_MICE_SMOTE_Filled.csv', index=False)
print("Unknown MICE+SMOTE Complete")