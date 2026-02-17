import os
import joblib
import numpy as np
from flask import Flask, render_template, request

# Create Flask app
app = Flask(__name__)

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "health_model.pkl")

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        try:
            fever = float(request.form["fever"])
            cough = float(request.form["cough"])
            fatigue = float(request.form["fatigue"])
            headache = float(request.form["headache"])

            # Make input array
            data = np.array([[fever, cough, fatigue, headache]])

            # Predict
            result = model.predict(data)[0]

            if result == 1:
                prediction = "⚠️ High Health Risk. Please consult a doctor."
            else:
                prediction = "✅ Low Health Risk. You seem fine."

        except Exception as e:
            prediction = "Error: " + str(e)

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(host="0.0.0.0", port=port, debug=True)
