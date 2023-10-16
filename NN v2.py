import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from custom_encoder import CustomLabelEncoder
from tensorflow.keras.models import load_model

data = pd.read_csv('Assignment3-Healthcare-Dataset.csv')

# Handle missing values
for column in data.columns:
    if data[column].dtype == 'object':
        data[column].fillna(data[column].mode()[0], inplace=True)
    else:
        data[column].fillna(data[column].median(), inplace=True)

# Convert categorical attributes to numerical using CustomLabelEncoder
label_encoders = {}
categorical_columns = data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split data into features (X) and target (y)
X = data.drop('ExpiredHospital', axis=1)
y = data['ExpiredHospital']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# # Define the model
# model = Sequential()
# model.add(Dense(units=128, activation='relu', input_shape=(X_train.shape[1],)))
# model.add(Dropout(0.5))
# model.add(Dense(units=64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(units=32, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(units=1, activation='sigmoid'))
#
# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# # Implement early stopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
#
# model_checkpoint = ModelCheckpoint("NN_Best_v3.h5", save_best_only=True, verbose=1)
#
# # Check the balance of the target variable and use class weights if imbalanced
# class_weights = {0: 1.0, 1: len(y) / y.sum() if y.sum() != 0 else 1.0}

# Train the model
# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping, model_checkpoint], class_weight=class_weights)

# # Evaluate the model
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Load your new dataset
new_data = pd.read_csv('Assignment3-Unknown-Dataset.csv')
model = load_model('NN_Best_v2.h5')

# Handle missing values (assuming median for numerical and mode for categorical)
for column in new_data.columns:
    if column in categorical_columns:
        new_data[column].fillna(new_data[column].mode()[0], inplace=True)
    else:
        new_data[column].fillna(new_data[column].median(), inplace=True)

# Encode categorical variables using the previously created label encoders
for column, encoder in label_encoders.items():
    if column in new_data.columns:
        # Get the mask for unseen labels
        unseen_mask = ~new_data[column].isin(encoder.classes_)

        # Temporarily replace unseen labels with a known label
        new_data.loc[unseen_mask, column] = encoder.classes_[0]

        # Now transform the data
        new_data[column] = encoder.transform(new_data[column])

# Scale the new data using the previously created scaler
new_data_scaled = scaler.transform(new_data.drop('ExpiredHospital', axis=1, errors='ignore'))

# Predict using the trained KNN classifier (assuming it's named knn_classifier)
predictions = model.predict(new_data_scaled)
binary_predictions = (predictions > 0.5).astype(int)

# Create a dataframe with RowID and Predicted-ExpiredHospital
output = pd.DataFrame({
    'row ID': ["Row" + str(idx) for idx in range(len(new_data))],
    'Predicted-ExpiredHospital': binary_predictions.flatten()
})

# Save the output to a CSV (optional)
output.to_csv('predictions_NN_v2.csv', index=False)