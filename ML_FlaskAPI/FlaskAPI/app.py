import pickle, traceback
import pandas as pd
from flask import Flask, json, request, jsonify

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/predict", methods=['GET'])
def predict_results():
    if model:
        try:
            df = pd.DataFrame(request.json)
            df = pd.get_dummies(df)
            df = df.reindex(columns=["Age","Sex_female","Sex_male","Sex_nan","Embarked_C","Embarked_Q","Embarked_S", "Embarked_nan"], fill_value=0)
            prediction = model.predict(df)
            return jsonify({"Prediction":str(prediction)})
        except:
            return jsonify({"Trace":traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')
if __name__ == "__main__":
    app.debug = True
    app.run("127.0.0.1", port=5000)