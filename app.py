from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])def predict():
        if request.method == 'GET':
            return render_template("index.html")

    step = float(request.form['step'])
    amount = float(request.form['amount'])
    oldbalanceOrg = float(request.form['oldbalanceOrg'])
    newbalanceOrig = float(request.form['newbalanceOrig'])
    oldbalanceDest = float(request.form['oldbalanceDest'])
    newbalanceDest = float(request.form['newbalanceDest'])

    txn_type = request.form['type']
    txn_type_encoded = encoder.transform([txn_type])[0]

    data = np.array([[step, txn_type_encoded, amount,
                      oldbalanceOrg, newbalanceOrig,
                      oldbalanceDest, newbalanceDest]])

    prediction = model.predict(data)[0]

    result = "Fraud Transaction" if prediction == 1 else "Legitimate Transaction"

    return render_template(
        "index.html",
        prediction_text=result,
        step=step,
        amount=amount,
        oldbalanceOrg=oldbalanceOrg,
        newbalanceOrig=newbalanceOrig,
        oldbalanceDest=oldbalanceDest,
        newbalanceDest=newbalanceDest,
        txn_type=txn_type
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 10000)))
