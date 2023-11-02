from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

class RotationForestClassifier:
    def __init__(self, n_estimators=10, n_features_per_subset=3, random_state=None):
        self.n_estimators = n_estimators
        self.n_features_per_subset = n_features_per_subset
        self.random_state = random_state
        self.trees = []
        self.rotation_matrices = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.trees = [DecisionTreeClassifier(random_state=self.random_state) for _ in range(self.n_estimators)]
        self.rotation_matrices = []

        for tree in self.trees:
            # Convert DataFrame to NumPy array if it's not already
            if isinstance(X, pd.DataFrame):
                X = X.to_numpy()

            # Randomly split the features into subsets
            feature_indices = np.random.permutation(n_features).reshape(-1, self.n_features_per_subset)

            rotation_matrix = np.zeros((n_features, n_features))
            for subset in feature_indices:
                # Apply PCA to the feature subset
                pca = PCA(random_state=self.random_state)
                X_subset = pca.fit_transform(X[:, subset])
                rotation_matrix[np.ix_(subset, subset)] = pca.components_

            # Rotate the entire feature space
            X_rotated = X.dot(rotation_matrix)

            # Train the decision tree on the rotated features
            tree.fit(X_rotated, y)

            # Store the rotation matrix
            self.rotation_matrices.append(rotation_matrix)

    def predict(self, X):
        # Apply the rotation and predict with each tree
        predictions = np.zeros((X.shape[0], self.n_estimators))
        for i, (tree, rotation_matrix) in enumerate(zip(self.trees, self.rotation_matrices)):
            X_rotated = X.dot(rotation_matrix)
            predictions[:, i] = tree.predict(X_rotated)

        # Majority voting
        final_predictions = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=predictions)
        return final_predictions


train_data = pd.read_csv('Dataset-Processed_Filled.csv')
pred_data = pd.read_csv('Dataset-Processed-Unknown_Filled.csv')

X = train_data.drop('ExpiredHospital', axis=1)
y = train_data['ExpiredHospital']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rotation_forest_clf = RotationForestClassifier(n_estimators=10, n_features_per_subset=5, random_state=42)
rotation_forest_clf.fit(X_train, y_train)

# Predict on the test set
rotation_forest_predictions = rotation_forest_clf.predict(X_test)

# Evaluate the Rotation Forest classifier
rotation_forest_report = classification_report(y_test, rotation_forest_predictions)

print(rotation_forest_report)

mlp_pred = rotation_forest_clf.predict(pred_data)

output = pd.DataFrame({
    'row ID': ["Row" + str(idx) for idx in pred_data.index],
    'Predicted-ExpiredHospital': mlp_pred.flatten()
})

output.to_csv('predictions_RotForest.csv', index=False)
