import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import random

# Create a mock dataset with 100 rows
np.random.seed(42)
data = {
    "IQ": np.random.normal(loc=110, scale=10, size=100).astype(int),
    "CGPA": np.round(np.random.uniform(6.0, 10.0, 100), 2),
    "10th Marks": np.round(np.random.uniform(60, 100, 100), 2),
    "12th Marks": np.round(np.random.uniform(60, 100, 100), 2),
    "Communication Skills": np.round(np.random.uniform(1, 10, 100), 1),
    "Placed": np.random.choice([0, 1], size=100, p=[0.1, 0.9])  # 0: Not Placed, 1: Placed
}

df = pd.DataFrame(data)

# Prepare features and labels
X = df.drop(columns=["Placed"])
y = df["Placed"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

import pickle

# Save the trained logistic regression model to a file
model_path = "model.pkl"
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)