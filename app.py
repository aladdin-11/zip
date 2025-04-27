from flask import Flask, render_template, request, redirect, session, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import numpy as np
import joblib

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Load trained model
model = joblib.load('cancer_predictor_rf_top15.pkl')

# In-memory user and history storage
users = {}
prediction_history = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        if username in users:
            return render_template('signup.html', error='Username already exists')

        users[username] = {'email': email, 'password': password}
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = users.get(username)
        if user and check_password_hash(user['password'], password):
            session['user'] = username
            return redirect(url_for('main'))
        else:
            return render_template('login.html', error='Invalid credentials')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/main')
def main():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('main.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # You can handle the contact form here
        return redirect(url_for('main'))
    return render_template('contact.html')

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))

    user_history = prediction_history.get(session['user'], [])
    return render_template('history.html', predictions=user_history)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            features = [
                float(request.form['PIK3CA_expr']),
                float(request.form['Pat_STK11_mutation']),
                float(request.form['HER2_expr']),
                float(request.form['BRCA1_expr']),
                float(request.form['KRAS_expr']),
                float(request.form['EGFR_expr']),
                float(request.form['TP53_expr']),
                float(request.form['Pat_KRAS_mutation']),
                float(request.form['Pat_TP53_mutation']),
                float(request.form['age']),
                float(request.form['Pat_Packs_Per_Year']),
                float(request.form['Pat_Smoking_Status']),
                float(request.form['CDH1_expr']),
                float(request.form['Pat_EGFR_mutation']),
                float(request.form['Pat_ALK_translocation'])
            ]

            input_array = np.array(features).reshape(1, -1)
            prediction = model.predict(input_array)[0]
            proba = model.predict_proba(input_array)[0]
            confidence = round(np.max(proba) * 100, 2)

            if prediction == 0:
                result = "Cancer Detected"
                cancer_type = "Breast"
            elif prediction == 1:
                result = "Cancer Detected"
                cancer_type = "Lung"
            else:
                result = "No Cancer Detected"
                cancer_type = "None"

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            history_entry = {
                'result': result,
                'cancer_type': cancer_type,
                'confidence': confidence,
                'timestamp': timestamp
            }

            prediction_history.setdefault(session['user'], []).append(history_entry)

            return render_template('predict.html', result=result, confidence=confidence, cancer_type=cancer_type)

        except Exception as e:
            return render_template('predict.html', result="Prediction Error", confidence=0, cancer_type="N/A")

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
