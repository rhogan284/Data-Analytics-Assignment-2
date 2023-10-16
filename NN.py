import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from custom_encoder import CustomLabelEncoder

data = pd.read_csv('Assignment3-Healthcare-Dataset.csv')

# Handle missing values by imputing the median for numerical columns and mode for categorical columns
for column in data.columns:
    if data[column].dtype == 'object':
        data[column].fillna(data[column].mode()[0], inplace=True)
    else:
        data[column].fillna(data[column].median(), inplace=True)

# Convert categorical attributes to numerical using CustomLabelEncoder
custom_label_encoders = {}
categorical_columns = data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    encoder = CustomLabelEncoder()
    data[column] = encoder.fit_transform(data[column])
    custom_label_encoders[column] = encoder

# Save the custom_label_encoders for future use
with open('custom_label_encoders.pkl', 'wb') as f:
    pickle.dump(custom_label_encoders, f)

# Split data into features (X) and target (y)
X = data.drop('ExpiredHospital', axis=1)
y = data['ExpiredHospital']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for future use
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(X_train.shape, X_test.shape)

# Define the model
model = Sequential()

# Input layer
model.add(Dense(units=128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.5))

# Hidden layers
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_checkpoint = ModelCheckpoint("NN_Best.h5", save_best_only=True, verbose=1)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[model_checkpoint])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
