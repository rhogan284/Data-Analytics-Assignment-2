import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('Dataset-Processed_NoMissing.csv')

# Split data into features (X) and target (y)
X = data.drop('ExpiredHospital', axis=1)
y = data['ExpiredHospital']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

optimized_rf_classifier = RandomForestClassifier(
    bootstrap=False,
    criterion='gini',
    max_depth=30,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=200
)
optimized_rf_classifier.fit(X_train_scaled, y_train)

predictions = optimized_rf_classifier.predict(X_test_scaled)

RF_accuracy = accuracy_score(y_test, predictions)
print(RF_accuracy)

# Load your new dataset
new_data = pd.read_csv('Dataset-Processed-Unknown_Missing.csv')

# Scale the new data using the previously created scaler
new_data_scaled = scaler.transform(new_data.drop('ExpiredHospital', axis=1, errors='ignore'))

# Predict using the trained KNN classifier (assuming it's named knn_classifier)
predictions = optimized_rf_classifier.predict(new_data_scaled)

# Create a dataframe with RowID and Predicted-ExpiredHospital
output = pd.DataFrame({
    'row ID': ["Row" + str(idx) for idx in new_data.index],
    'Predicted-ExpiredHospital': predictions
})

# Save the output to a CSV (optional)
output.to_csv('predictions_RF_optimised_v3.csv', index=False)

