from flask import Flask, render_template, request
import numpy as np
import joblib
import traceback

app = Flask(__name__)

# Load models and artifacts
rf = joblib.load("model_rf.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
threshold = joblib.load("threshold.pkl")

# ==============================
# VALIDATION FUNCTION
# ==============================
def validate_inputs(step, amount, oldbalanceOrg, newbalanceOrig,
                    oldbalanceDest, newbalanceDest, txn_type):
    errors = []
    if amount < 0:
        errors.append("Amount cannot be negative")
    if oldbalanceOrg < 0:
        errors.append("Sender balance cannot be negative")
    if newbalanceOrig < 0:
        errors.append("Sender balance after cannot be negative")
    if oldbalanceDest < 0:
        errors.append("Receiver balance cannot be negative")
    if newbalanceDest < 0:
        errors.append("Receiver balance after cannot be negative")
    if step < 0 or step > 744:
        errors.append("Step must be between 0 and 744")
    if txn_type not in ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT']:
        errors.append("Invalid transaction type")
    return errors

# ==============================
# RULE-BASED OVERRIDES
# ==============================
def apply_rule_overrides(prob, amount, amount_ratio, is_empty_receiver,
                         oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest):
    if amount_ratio > 0.9:
        prob = max(prob, 0.8)
    if is_empty_receiver == 1 and amount > 100000:
        prob = max(prob, 0.85)
    if amount > 1000000:
        prob = 1.0
    if (oldbalanceOrg - newbalanceOrig) != (newbalanceDest - oldbalanceDest):
        prob = max(prob, 0.9)
    if newbalanceOrig < 0 or newbalanceDest < 0:
        prob = max(prob, 0.8)
    suspicious_amounts = [4900, 4999, 9900, 9999, 1999, 2999, 3999, 4999, 5999, 6999, 7999, 8999, 9999]
    if amount in suspicious_amounts and amount_ratio > 0.5:
        prob = max(prob, 0.7)
    return prob

@app.route('/')
def home():
    # No POST → fresh GET → empty form
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        step = float(request.form['step'])
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])
        txn_type = request.form['type']

        # Validation
        errors = validate_inputs(step, amount, oldbalanceOrg, newbalanceOrig,
                                 oldbalanceDest, newbalanceDest, txn_type)
        if errors:
            # Keep the entered values and show error as fraud
            return render_template(
                "index.html",
                prediction_text=f"❌ Fraud: {', '.join(errors)}",
                result_class="fraud",
                step=step, amount=amount,
                oldbalanceOrg=oldbalanceOrg, newbalanceOrig=newbalanceOrig,
                oldbalanceDest=oldbalanceDest, newbalanceDest=newbalanceDest,
                txn_type=txn_type
            )

        # ---- Feature engineering (same as training) ----
        amount_ratio = amount / (oldbalanceOrg + 1)
        balance_error = abs((oldbalanceOrg - newbalanceOrig) - (newbalanceDest - oldbalanceDest))
        is_empty_receiver = 1 if oldbalanceDest == 0 else 0
        balance_mismatch = 1 if (oldbalanceOrg - newbalanceOrig) != (newbalanceDest - oldbalanceDest) else 0
        sender_negative = 1 if newbalanceOrig < 0 else 0
        receiver_negative = 1 if newbalanceDest < 0 else 0

        hour = step % 24
        unusual_hour = 1 if (hour >= 23 or hour <= 6) else 0

        high_percentage_transfer = 1 if (amount_ratio > 0.9 and txn_type == 'TRANSFER') else 0
        round_amount = 1 if amount % 100 == 0 else 0

        suspicious_amounts = [4900, 4999, 9900, 9999, 1999, 2999, 3999, 4999, 5999, 6999, 7999, 8999, 9999]
        suspicious_amount = 1 if amount in suspicious_amounts else 0

        amount_to_dest_ratio = amount / (oldbalanceDest + 1)
        unusual_hour_amount = unusual_hour * amount
        unusual_hour_ratio = unusual_hour * amount_ratio
        empty_receiver_high_amount = is_empty_receiver * (1 if amount > 100000 else 0)

        # Type dummies
        type_cash_out = 1 if txn_type == 'CASH_OUT' else 0
        type_debit = 1 if txn_type == 'DEBIT' else 0
        type_payment = 1 if txn_type == 'PAYMENT' else 0
        type_transfer = 1 if txn_type == 'TRANSFER' else 0

        # Build feature dictionary using saved feature names
        values = {}
        for name in feature_names:
            if name == 'step':
                values[name] = step
            elif name == 'amount':
                values[name] = amount
            elif name == 'oldbalanceOrg':
                values[name] = oldbalanceOrg
            elif name == 'newbalanceOrig':
                values[name] = newbalanceOrig
            elif name == 'oldbalanceDest':
                values[name] = oldbalanceDest
            elif name == 'newbalanceDest':
                values[name] = newbalanceDest
            elif name == 'amount_ratio':
                values[name] = amount_ratio
            elif name == 'balance_error':
                values[name] = balance_error
            elif name == 'is_empty_receiver':
                values[name] = is_empty_receiver
            elif name == 'balance_mismatch':
                values[name] = balance_mismatch
            elif name == 'sender_negative':
                values[name] = sender_negative
            elif name == 'receiver_negative':
                values[name] = receiver_negative
            elif name == 'unusual_hour':
                values[name] = unusual_hour
            elif name == 'high_percentage_transfer':
                values[name] = high_percentage_transfer
            elif name == 'round_amount':
                values[name] = round_amount
            elif name == 'suspicious_amount':
                values[name] = suspicious_amount
            elif name == 'amount_to_dest_ratio':
                values[name] = amount_to_dest_ratio
            elif name == 'unusual_hour_amount':
                values[name] = unusual_hour_amount
            elif name == 'unusual_hour_ratio':
                values[name] = unusual_hour_ratio
            elif name == 'empty_receiver_high_amount':
                values[name] = empty_receiver_high_amount
            elif name == 'type_CASH_OUT':
                values[name] = type_cash_out
            elif name == 'type_DEBIT':
                values[name] = type_debit
            elif name == 'type_PAYMENT':
                values[name] = type_payment
            elif name == 'type_TRANSFER':
                values[name] = type_transfer

        # Order features and scale
        input_data = [values[name] for name in feature_names]
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        # Predict probability
        prob = rf.predict_proba(input_scaled)[0][1]

        # Apply rule overrides
        prob = apply_rule_overrides(prob, amount, amount_ratio, is_empty_receiver,
                                    oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest)

        # Final decision
        prediction = 1 if prob >= threshold else 0
        result_text = "🚨 Fraud Transaction" if prediction == 1 else "✅ Legitimate Transaction"
        result_text += f" (Fraud probability: {prob:.2f})"
        result_class = "fraud" if prediction == 1 else "legit"

        # Render with the same input values so the form retains them
        return render_template(
            "index.html",
            prediction_text=result_text,
            result_class=result_class,
            step=step, amount=amount,
            oldbalanceOrg=oldbalanceOrg, newbalanceOrig=newbalanceOrig,
            oldbalanceDest=oldbalanceDest, newbalanceDest=newbalanceDest,
            txn_type=txn_type
        )

    except Exception as e:
        print("Error in /predict:")
        traceback.print_exc()
        # Still try to keep entered values (if any)
        return render_template(
            "index.html",
            prediction_text=f"❌ Fraud: Error: {str(e)}",
            result_class="fraud",
            step=request.form.get('step', ''),
            amount=request.form.get('amount', ''),
            oldbalanceOrg=request.form.get('oldbalanceOrg', ''),
            newbalanceOrig=request.form.get('newbalanceOrig', ''),
            oldbalanceDest=request.form.get('oldbalanceDest', ''),
            newbalanceDest=request.form.get('newbalanceDest', ''),
            txn_type=request.form.get('type', '')
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
