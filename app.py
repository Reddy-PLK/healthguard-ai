from flask import Flask, render_template, request
import joblib
import os
import numpy as np
model = joblib.load("model/health_model.pkl")
app = Flask(__name__)

# Load trained model
MODEL_PATH = os.path.join("model", "health_model.pkl")   # change name if different
model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user inputs
        fever = float(request.form["fever"])
        cough = float(request.form["cough"])
        fatigue = float(request.form["fatigue"])
        headache = float(request.form["headache"])

        # Prepare input
        input_data = np.array([[fever, cough, fatigue, headache]])

        # Predict
        result = model.predict(input_data)[0]

        return render_template(
            "index.html",
            prediction=f"Health Risk Level: {result}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction="Error: " + str(e)
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)