from flask import Flask, jsonify, render_template, request
import pickle
import numpy as np
app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

from features import FEATURES
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET'])
def predict():
    if not request.json:
        return jsonify({"error":"no request recieved" })
    x_dict, missing_flag = parse_args(request.json)
    x_array = np.array([x_dict])
    estimate = int(model.predict(x_array)[0])
    response = dict(ESTIMATE=estimate, MISSING_FLAG=missing_flag)
    return jsonify(response)
    
def parse_args(request_dict):
    x_dict = []
    missing_flag = False
    for feature in FEATURES:
        value = request_dict.get(feature)
        if value:
            x_dict.append(value)
        else:
            missing_flag = True
            x_dict.append(0)
    return x_dict, missing_flag

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)