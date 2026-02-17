import os
import joblib
import numpy as np
from flask import Flask, render_template, request

# Create Flask app
app = Flask(__name__)

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model
MODEL_PATH = os.path.join(BASE_DIR, "model", "health_model.pkl")

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":
        try:
            # Get inputs from form
            fever = int(request.form["fever"])
            cough = int(request.form["cough"])
            headache = int(request.form["headache"])
            breathing = int(request.form["breathing"])
            fatigue = int(request.form["fatigue"])

            # Create input array
            input_data = np.array(
                [[fever, cough, headache, breathing, fatigue]]
            )

            # Predict
            result = model.predict(input_data)

            prediction = result[0]

        except Exception as e:
            prediction = "Error: " + str(e)

    return render_template("index.html", prediction=prediction)


# Run app (for Render + local)
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(host="0.0.0.0", port=port, debug=True)
