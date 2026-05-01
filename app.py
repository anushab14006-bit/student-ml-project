from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# load model once
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # get values from request
    G1 = data['G1']
    G2 = data['G2']
    studytime = data['studytime']

    # create dataframe
    sample = pd.DataFrame([[G1, G2, studytime]],
                          columns=['G1', 'G2', 'studytime'])

    result = model.predict(sample)[0]

    return jsonify({'prediction': int(result)})

if __name__ == '__main__':
    app.run(debug=True)