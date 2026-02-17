import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ---------------- PATHS ----------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(BASE_DIR, "symptoms.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "health_model.pkl")


# ---------------- LOAD DATA ----------------

data = pd.read_csv(CSV_PATH)

print("Dataset Loaded Successfully")
print("Columns:", data.columns)


# ---------------- FEATURES & TARGET ----------------

X = data[[
    "fever",
    "cough",
    "headache",
    "breathing_problem",
    "fatigue"
]]

y = data["disease"]


# ---------------- SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ---------------- TRAIN ----------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# ---------------- TEST ----------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy*100:.2f}%")


# ---------------- SAVE ----------------

joblib.dump(model, MODEL_PATH)

print("Model Saved At:", MODEL_PATH)
print("Training Completed Successfully")
