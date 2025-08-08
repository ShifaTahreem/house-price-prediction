from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        try:
            # Get values from form
            gr_liv_area = float(request.form['gr_liv_area'])
            total_bsmt_sf = float(request.form['total_bsmt_sf'])
            garage_area = float(request.form['garage_area'])
            full_bath = int(request.form['full_bath'])
            overall_qual = int(request.form['overall_qual'])

            # Make prediction
            input_data = np.array([[gr_liv_area, total_bsmt_sf, garage_area, full_bath, overall_qual]])
            prediction = model.predict(input_data)[0]

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
