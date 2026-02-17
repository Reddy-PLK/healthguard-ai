import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier


# Get root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# File paths
CSV_PATH = os.path.join(BASE_DIR, "symptoms.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "health_model.pkl")


# Load dataset
data = pd.read_csv(CSV_PATH)

print("Dataset Columns:")
print(data.columns)


# Select required features
X = data[[
    "fever",
    "cough",
    "headache",
    "breathing_problem",
    "fatigue"
]]

y = data["disease"]


# Train model
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)

model.fit(X, y)


# Save model
joblib.dump(model, MODEL_PATH)

print("Model trained successfully!")
print("Saved at:", MODEL_PATH)
