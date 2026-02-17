from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model
model = joblib.load("model/health_model.pkl")


# Home Page
@app.route("/")
def home():
    return render_template("home.html")


# Prediction Page (Form)
@app.route("/check")
def check():
    return render_template("predict.html")


# About Page
@app.route("/about")
def about():
    return render_template("about.html")


# Contact Page
@app.route("/contact")
def contact():
    return render_template("contact.html")


# Prediction Logic
@app.route("/predict", methods=["POST"])
def predict():

    fever = float(request.form["fever"])
    cough = float(request.form["cough"])
    fatigue = float(request.form["fatigue"])
    headache = float(request.form["headache"])
    breathing = float(request.form["breathing"])

    data = np.array([[fever, cough, fatigue, headache, breathing]])

    result = model.predict(data)[0]

    if result == 1:
        output = "⚠️ High Health Risk! Please consult a doctor."
    else:
        output = "✅ Low Health Risk. You are doing well!"

    return render_template("predict.html", prediction=output)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
