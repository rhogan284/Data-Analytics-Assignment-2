import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('Assignment3-Healthcare-Dataset.csv')

# Handle missing values by imputing the median for numerical columns and mode for categorical columns
for column in data.columns:
    if data[column].dtype == 'object':
        data[column].fillna(data[column].mode()[0], inplace=True)
    else:
        data[column].fillna(data[column].median(), inplace=True)

# Convert categorical attributes to numerical using label encoding
label_encoders = {}
categorical_columns = data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split data into features (X) and target (y)
X = data.drop('ExpiredHospital', axis=1)
y = data['ExpiredHospital']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train K-Nearest Neighbors Classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train_scaled, y_train)

# Predict on test set
knn_predictions = knn_classifier.predict(X_test_scaled)

# Evaluate accuracy
knn_accuracy = accuracy_score(y_test, knn_predictions)
print(knn_accuracy)


# Load your new dataset
new_data = pd.read_csv('Assignment3-Unknown-Dataset.csv')

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
predictions = knn_classifier.predict(new_data_scaled)

# Create a dataframe with RowID and Predicted-ExpiredHospital
output = pd.DataFrame({
    'row ID': ["Row" + str(idx) for idx in new_data.index],
    'Predicted-ExpiredHospital': predictions
})

# Save the output to a CSV (optional)
output.to_csv('predictions_KNN.csv', index=False)