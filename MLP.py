from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

train_data = pd.read_csv('Dataset-Processed_Filled.csv')
pred_data = pd.read_csv('Dataset-Processed-Unknown_Filled.csv')

X = train_data.drop('ExpiredHospital', axis=1)
y = train_data['ExpiredHospital']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

optimised_mlp_clf = MLPClassifier(
    activation='relu',
    hidden_layer_sizes=(50,),
    learning_rate_init=0.01,
    max_iter=300,
    random_state=42
)
optimised_mlp_clf.fit(X_train, y_train)

# Predict on the test set
optimized_mlp_predictions = optimised_mlp_clf.predict(X_test)

# Evaluate the optimized MLP classifier
optimized_mlp_report = classification_report(y_test, optimized_mlp_predictions)

print(optimized_mlp_report)

mlp_pred = optimised_mlp_clf.predict(pred_data)

output = pd.DataFrame({
    'row ID': ["Row" + str(idx) for idx in pred_data.index],
    'Predicted-ExpiredHospital': mlp_pred.flatten()
})

output.to_csv('predictions_MLP_v2.csv', index=False)
