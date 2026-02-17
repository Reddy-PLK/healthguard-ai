import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
CSV_PATH = os.path.join(BASE_DIR, "symptoms.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "health_model.pkl")


# Load dataset
data = pd.read_csv(CSV_PATH)

print("Dataset columns:")
print(data.columns)


# Features and target
X = data[["fever", "cough", "headache", "breathing_problem", "fatigue"]]
y = data["disease"]


# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# Test accuracy
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")


# Save model
joblib.dump(model, MODEL_PATH)

print("Model saved at:", MODEL_PATH)
print("Training complete!")
