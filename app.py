from flask import Flask, render_template, request
import joblib
import os


app = Flask(__name__)


# Load trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "health_model.pkl")

model = joblib.load(MODEL_PATH)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    try:
        fever = float(request.form["fever"])
        cough = float(request.form["cough"])
        headache = float(request.form["headache"])
        breathing = float(request.form["breathing"])
        fatigue = float(request.form["fatigue"])

        # Input in same order as training
        input_data = [[
            fever,
            cough,
            headache,
            breathing,
            fatigue
        ]]

        prediction = model.predict(input_data)[0]

        return render_template(
            "index.html",
            prediction=f"Predicted Disease: {prediction}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}"
        )


# Run server
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
