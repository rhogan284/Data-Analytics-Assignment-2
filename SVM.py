from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('Dataset-Processed_Filled.csv')
pred_data = pd.read_csv('Dataset-Processed-Unknown_Filled.csv')

X = train_data.drop('ExpiredHospital', axis=1)
y = train_data['ExpiredHospital']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_clf = SVC(kernel='poly', random_state=42)
svm_clf.fit(X_train, y_train)

svm_predictions = svm_clf.predict(X_test)

svm_report = classification_report(y_test, svm_predictions)

print(svm_report)