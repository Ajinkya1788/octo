import numpy as np
from flask import Flask, request, render_template, json, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open('house_price_model_pickle.pkl', 'rb'))

with open('columns.json', 'r') as f:
    __data_columns = json.load(f)['data_columns']

data_columns = np.array(__data_columns, dtype="object")


def predict_price(location, size, total_sqft, balcony):
    loc_index = np.where(data_columns == location)[0][0]
    x = np.zeros(len(data_columns))
    x[0] = size
    x[1] = total_sqft
    x[2] = balcony
    if loc_index >= 0:
        x[loc_index] = 1
    return round(model.predict([x])[0], 2)


@app.route("/home")
def home():
    return render_template('index.html')


@app.route('/predict', methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        total_sqft = float(request.form['total_sqft'])
        location = str(request.form.get('location'))
        size = int(request.form['bhk'])
        balcony = int(request.form['balcony'])
        output = predict_price(location, size, total_sqft, balcony)
        return render_template('index.html', prediction_text='approx price in Lacs will be Rs{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
