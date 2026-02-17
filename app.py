from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("model/health_model.pkl")


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

        # Get form data
        fever = float(request.form.get("fever"))
        cough = float(request.form.get("cough"))
        headache = float(request.form.get("headache"))
        breathing = float(request.form.get("breathing"))
        fatigue = float(request.form.get("fatigue"))

        # IMPORTANT: Same order as training
        data = np.array([[
            fever,
            cough,
            headache,
            breathing,
            fatigue
        ]])

        # Predict
        result = model.predict(data)[0]

        # Message
        if result == 1:
            output = "⚠️ High Health Risk! Please consult a doctor."
            status = "danger"
        else:
            output = "✅ Low Health Risk. You are doing well!"
            status = "safe"

        # Values for chart
        values = [
            fever,
            cough,
            headache,
            breathing,
            fatigue
        ]

        return render_template(
            "predict.html",
            prediction=output,
            status=status,
            values=values
        )

    except Exception as e:

        return render_template(
            "predict.html",
            prediction="❌ Error: " + str(e)
        )


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
