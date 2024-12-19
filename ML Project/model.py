import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
diabetes = pd.read_csv("diabetes.csv")

# Prepare the data
X = diabetes.drop(columns="Outcome", axis=1)
y = diabetes["Outcome"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=2)

# Train the SVM model
model = svm.SVC(kernel="linear")
model.fit(X_train, y_train)

# Evaluate the model
train_predicted = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predicted)
print(f"Training Accuracy: {train_accuracy:.2f}")

test_predicted = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predicted)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Save the model and scaler
with open("diabetes_model.pkl", "wb") as model_file: 
    pickle.dump(model, model_file)                   

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully!")



