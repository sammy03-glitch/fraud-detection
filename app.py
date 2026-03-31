import pandas as pd
import os
from dotenv import load_dotenv
from flask import Flask, request, session, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import bcrypt
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import io

# ─────────────────────────────────────────
#  APP SETUP
# ─────────────────────────────────────────
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# CORS lets your Android app talk to Flask from a different device/IP
CORS(app)

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# ─────────────────────────────────────────
#  DATABASE CONNECTION
# ─────────────────────────────────────────
client = MongoClient(MONGO_URI)
db = client['user_database']
collection = db['users']


# ─────────────────────────────────────────
#  HELPER — build a clean JSON response
# ─────────────────────────────────────────
def resp(data: dict, status: int = 200):
    return jsonify(data), status


# ═══════════════════════════════════════════════════════
#  ROUTE 1 — Home
# ═══════════════════════════════════════════════════════
@app.route('/')
def index():
    return resp({"message": "Credit Card Fraud Detection API is running!"})


# ═══════════════════════════════════════════════════════
#  ROUTE 2 — Register
#  METHOD: POST
#  BODY (JSON): { "username": "...", "email": "...", "password": "..." }
# ═══════════════════════════════════════════════════════
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    if not data:
        return resp({"status": "error", "message": "No data received"}, 400)

    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    email    = data.get('email', '').strip()

    if not username or not password or not email:
        return resp({"status": "error", "message": "Missing fields"}, 400)

    if collection.find_one({'username': username}):
        return resp({"status": "error", "message": "Username already exists"}, 409)

    if collection.find_one({'email': email}):
        return resp({"status": "error", "message": "Email already exists"}, 409)

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    collection.insert_one({
        'username': username,
        'password': hashed_password,
        'email':    email
    })

    return resp({"status": "success", "message": "User registered successfully"}, 201)


# ═══════════════════════════════════════════════════════
#  ROUTE 3 — Login
#  METHOD: POST
#  BODY (JSON): { "username": "...", "password": "..." }
# ═══════════════════════════════════════════════════════
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()

    if not data:
        return resp({"status": "error", "message": "No data received"}, 400)

    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    if not username or not password:
        return resp({"status": "error", "message": "Missing fields"}, 400)

    # Dummy login for quick testing
    if username == 'dummy' and password == 'dummy':
        return resp({"status": "success", "username": "Demo User"})

    user = collection.find_one({'username': username})

    if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return resp({"status": "success", "username": user['username']})
    else:
        return resp({"status": "error", "message": "Invalid username or password"}, 401)


# ═══════════════════════════════════════════════════════
#  ROUTE 4 — Predict (CSV upload)
#  METHOD: POST
#  BODY: multipart/form-data → key="file", value=<CSV file>
# ═══════════════════════════════════════════════════════
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return resp({"status": "error", "message": "No file uploaded"}, 400)

    file = request.files['file']

    if file.filename == '':
        return resp({"status": "error", "message": "No file selected"}, 400)

    try:
        content = file.read()
        data = pd.read_csv(io.BytesIO(content))

        if 'Class' not in data.columns:
            return resp({
                "status": "error",
                "message": "CSV must have a 'Class' column"
            }, 400)

        total_transactions   = len(data)
        fraudulent_count     = int((data['Class'] == 1).sum())
        non_fraudulent_count = int((data['Class'] == 0).sum())

        print(f"Dataset loaded: {total_transactions} rows, {fraudulent_count} fraud cases")

        fraud_data  = data[data['Class'] == 1]
        normal_data = data[data['Class'] == 0].sample(
                        n=min(9508, len(data[data['Class'] == 0])),
                        random_state=42)
        sample = pd.concat([fraud_data, normal_data]).sample(
                        frac=1, random_state=42)

        X = sample.drop(columns=["Class"])
        y = sample["Class"]

        print(f"Training on {len(sample)} rows...")

        # Isolation Forest
        print("Training Isolation Forest...")
        iso = IsolationForest(random_state=42, contamination=0.05)
        iso.fit(X)
        iso_preds  = [0 if p == 1 else 1 for p in iso.predict(X)]
        iso_acc    = round(accuracy_score(y, iso_preds) * 100, 2)
        iso_report = classification_report(y, iso_preds, output_dict=True)
        print(f"Isolation Forest done — Accuracy: {iso_acc}%")

        # Logistic Regression
        print("Training Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X, y)
        lr_preds  = lr.predict(X)
        lr_acc    = round(accuracy_score(y, lr_preds) * 100, 2)
        lr_report = classification_report(y, lr_preds, output_dict=True)
        print(f"Logistic Regression done — Accuracy: {lr_acc}%")

        # SVM
        print("Training SVM...")
        svm_sample = sample.sample(n=min(2000, len(sample)), random_state=42)
        X_s = svm_sample.drop(columns=["Class"])
        y_s = svm_sample["Class"]
        svm = SVC(random_state=42)
        svm.fit(X_s, y_s)
        svm_preds  = svm.predict(X_s)
        svm_acc    = round(accuracy_score(y_s, svm_preds) * 100, 2)
        svm_report = classification_report(y_s, svm_preds, output_dict=True)
        print(f"SVM done — Accuracy: {svm_acc}%")

        print("All models done! Sending results...")

        return resp({
            "status": "success",
            "summary": {
                "total_transactions":   total_transactions,
                "fraudulent_count":     fraudulent_count,
                "non_fraudulent_count": non_fraudulent_count
            },
            "models": {
                "isolation_forest": {
                    "accuracy_percent": iso_acc,
                    "precision": round(iso_report.get('1', {}).get('precision', 0), 4),
                    "recall":    round(iso_report.get('1', {}).get('recall', 0), 4),
                    "f1_score":  round(iso_report.get('1', {}).get('f1-score', 0), 4)
                },
                "logistic_regression": {
                    "accuracy_percent": lr_acc,
                    "precision": round(lr_report.get('1', {}).get('precision', 0), 4),
                    "recall":    round(lr_report.get('1', {}).get('recall', 0), 4),
                    "f1_score":  round(lr_report.get('1', {}).get('f1-score', 0), 4)
                },
                "svm": {
                    "accuracy_percent": svm_acc,
                    "note": "Trained on 2000-row sample for speed",
                    "precision": round(svm_report.get('1', {}).get('precision', 0), 4),
                    "recall":    round(svm_report.get('1', {}).get('recall', 0), 4),
                    "f1_score":  round(svm_report.get('1', {}).get('f1-score', 0), 4)
                }
            }
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return resp({"status": "error", "message": str(e)}, 500)


# ═══════════════════════════════════════════════════════
#  ROUTE 5 — Logout
#  METHOD: POST
# ═══════════════════════════════════════════════════════
@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return resp({"status": "success", "message": "Logged out"})


# ═══════════════════════════════════════════════════════
#  ROUTE 6 — Predict Single Transaction
#  METHOD: POST
#  BODY (JSON): { "amount": 150.0, "time": 50000, "v1": -1.35, ... }
# ═══════════════════════════════════════════════════════
@app.route('/predict_single', methods=['POST'])
def predict_single():
    try:
        data = request.get_json()
        if not data:
            return resp({"status": "error", "message": "No data received"}, 400)

        sample_path = os.path.join('data', 'creditcard.csv')

        if not os.path.exists(sample_path):
            return resp({
                "status": "error",
                "message": "Training data not found on server."
            }, 400)

        print("Loading dataset for single prediction...")
        train_df = pd.read_csv(sample_path)

        fraud_rows  = train_df[train_df['Class'] == 1]
        normal_rows = train_df[train_df['Class'] == 0].sample(
                        n=len(fraud_rows), random_state=42)
        sample = pd.concat([fraud_rows, normal_rows]).sample(
                        frac=1, random_state=42)

        X_train = sample.drop(columns=['Class'])
        y_train = sample['Class']

        row = {}
        row['Time']   = float(data.get('time',   50000))
        row['Amount'] = float(data.get('amount', 0.0))

        for i in range(1, 29):
            key_lower = f'v{i}'
            key_upper = f'V{i}'
            if key_lower in data:
                row[key_upper] = float(data[key_lower])
            elif key_upper in data:
                row[key_upper] = float(data[key_upper])
            else:
                row[key_upper] = 0.0

        X_input = pd.DataFrame([row])[X_train.columns]

        print(f"Input: Amount={row['Amount']}, Time={row['Time']}")
        print(f"Training on {len(sample)} balanced rows...")

        results = {}

        # Isolation Forest
        print("Running Isolation Forest...")
        iso = IsolationForest(n_estimators=100, contamination=0.5, random_state=42)
        iso.fit(X_train)
        iso_pred  = iso.predict(X_input)[0]
        iso_score = iso.decision_function(X_input)[0]
        iso_label = "FRAUD" if iso_pred == -1 else "SAFE"
        iso_conf  = min(99.9, max(50.0, round(float(abs(iso_score) * 100), 1)))
        results['isolation_forest'] = {"prediction": iso_label, "confidence": iso_conf}
        print(f"Isolation Forest: {iso_label}")

        # Logistic Regression
        print("Running Logistic Regression...")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_input_scaled = scaler.transform(X_input)

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_scaled, y_train)
        lr_pred  = lr.predict(X_input_scaled)[0]
        lr_proba = lr.predict_proba(X_input_scaled)[0]
        lr_label = "FRAUD" if lr_pred == 1 else "SAFE"
        lr_conf  = round(float(max(lr_proba)) * 100, 1)
        results['logistic_regression'] = {"prediction": lr_label, "confidence": lr_conf}
        print(f"Logistic Regression: {lr_label}")

        # SVM
        print("Running SVM...")
        svm = SVC(probability=True, kernel='rbf', random_state=42)
        svm.fit(X_train_scaled, y_train)
        svm_pred  = svm.predict(X_input_scaled)[0]
        svm_proba = svm.predict_proba(X_input_scaled)[0]
        svm_label = "FRAUD" if svm_pred == 1 else "SAFE"
        svm_conf  = round(float(max(svm_proba)) * 100, 1)
        results['svm'] = {"prediction": svm_label, "confidence": svm_conf}
        print(f"SVM: {svm_label}")

        fraud_votes = sum(1 for m in results.values() if m['prediction'] == 'FRAUD')
        overall = "FRAUD" if fraud_votes >= 2 else "SAFE"
        print(f"Final verdict: {overall} ({fraud_votes}/3 votes)")

        return resp({
            "status":      "success",
            "overall":     overall,
            "fraud_votes": fraud_votes,
            "models":      results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return resp({"status": "error", "message": str(e)}, 500)


# ─────────────────────────────────────────
#  START THE SERVER
# ─────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)