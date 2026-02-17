from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os


app = Flask(__name__)


# ---------------- LOAD MODEL ----------------

MODEL_PATH = os.path.join("model", "health_model.pkl")
model = joblib.load(MODEL_PATH)


# ---------------- HOME ----------------

@app.route("/")
def home():
    return render_template("home.html")


# ---------------- CHECK PAGE ----------------

@app.route("/check")
def check():
    return render_template("predict.html")


# ---------------- ABOUT ----------------

@app.route("/about")
def about():
    return render_template("about.html")


# ---------------- CONTACT ----------------

@app.route("/contact")
def contact():
    return render_template("contact.html")


# ---------------- PREDICT ----------------

@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get inputs
        fever = float(request.form["fever"])
        cough = float(request.form["cough"])
        headache = float(request.form["headache"])
        breathing = float(request.form["breathing_problem"])
        fatigue = float(request.form["fatigue"])

        # Create DataFrame (safe input)
        input_data = pd.DataFrame([{
            "fever": fever,
            "cough": cough,
            "headache": headache,
            "breathing_problem": breathing,
            "fatigue": fatigue
        }])

        # Predict
        result = model.predict(input_data)[0]

        # Output
        if result == 1:
            output = "⚠️ High Health Risk! Please consult a doctor."
            status = "danger"
        else:
            output = "✅ Low Health Risk. You are doing well!"
            status = "safe"

        values = [fever, cough, headache, breathing, fatigue]

        return render_template(
            "predict.html",
            prediction=output,
            status=status,
            values=values
        )

    except Exception as e:

        print("ERROR:", e)

        return render_template(
            "predict.html",
            prediction="❌ Please fill all fields correctly!"
        )


# ---------------- RUN SERVER ----------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
