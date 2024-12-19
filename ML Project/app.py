from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the model and scaler
with open("diabetes_model.pkl", "rb") as model_file:
    diabetes_model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = float(request.form['age'])

        input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]).reshape(1, -1)

        scaled_data = scaler.transform(input_data)

        prediction = diabetes_model.predict(scaled_data)
        result = "The patient is diabetic" if prediction[0] == 1 else "The patient is non-diabetic"

        return render_template('index.html', result=result)

    except Exception as e:
        error_message = f"Error: {str(e)}"
        return render_template('index.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
