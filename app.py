from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and the scaler used during training
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    cgpa = float(request.form['cgpa'])
    iq = float(request.form['iq'])

    # scale inputs exactly the way they were scaled during training
    scaled_input = scaler.transform(np.array([[cgpa, iq]]))

    prediction = model.predict(scaled_input)[0]
    confidence = model.predict_proba(scaled_input)[0][prediction] * 100

    if prediction == 1:
        message = "Ho jayega Placement!"
        css_class = "result-yes"
    else:
        message = "Beta Tumse na ho payega!"
        css_class = "result-no"

    return render_template(
        'index.html',
        prediction=message,
        css_class=css_class,
        confidence=round(confidence, 1),
        cgpa=cgpa,
        iq=iq
    )


if __name__ == '__main__':
    app.run(debug=True)
