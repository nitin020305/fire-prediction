from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("rf.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        features = [
            float(request.form['Temperature']),
            float(request.form['RH']),
            float(request.form['Ws']),
            float(request.form['Rain']),
            float(request.form['FFMC']),
            float(request.form['DMC']),
            float(request.form['ISI']),
            float(request.form['Region'])
        ]

        # Convert to numpy
        final_input = np.array([features])

        # Scale input
        scaled_input = scaler.transform(final_input)

        # Predict probability
        proba = model.predict_proba(scaled_input)
        fire_probability = proba[0][1] * 100

        if fire_probability > 60:
            risk_level = "High Fire Risk 🔥"
        else:
            risk_level = "Low Fire Risk 🌧"

        return render_template(
            "index.html",
            prediction_text=f"Chance of Fire: {fire_probability:.2f}%",
            risk=risk_level
        )

    except:
        return render_template("index.html", prediction_text="Invalid Input!")

if __name__ == "__main__":
    app.run(debug=True)