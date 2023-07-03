from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

# Import ridge regressor model and standard scaler pickle files
standard_scaler = pickle.load(open("models/scaler.pkl", "rb"))
reg_model = pickle.load(open("models/ridge.pkl", "rb"))

# Route for the home page
@app.route("/", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Get the input values from the form
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # Scale the input data using the pre-trained scaler
        new_scaled_data = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

        # Make the prediction using the pre-trained regression model
        result = reg_model.predict(new_scaled_data)

        # Render the template with the result
        return render_template("index.html", result=result[0])
    else:
        # Render the template for the home page
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")
