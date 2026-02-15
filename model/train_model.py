import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "symptoms.csv")

data = pd.read_csv(csv_path)
print(data.columns)

# Split features and target
X = data[['fever', 'cough', 'headache', 'fatigue']]
y = data["disease"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "model/health_model.pkl")

print("Model trained and saved successfully!")