import pandas as pd
import os
import gc
from dotenv import load_dotenv
from flask import Flask, request, session, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import bcrypt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import io

# ─────────────────────────────────────────
#  APP SETUP
# ─────────────────────────────────────────
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
CORS(app)

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# ─────────────────────────────────────────
#  LAZY DATABASE CONNECTION
#  Only connects when first needed
#  Saves memory on startup
# ─────────────────────────────────────────
_client = None

def get_collection():
    global _client
    if _client is None:
        print("Connecting to MongoDB...")
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
        print("MongoDB connected!")
    return _client['user_database']['users']


# ─────────────────────────────────────────
#  HELPER
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
# ═══════════════════════════════════════════════════════
@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        if not data:
            return resp({"status": "error", "message": "No data received"}, 400)

        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        email    = data.get('email', '').strip()

        if not username or not password or not email:
            return resp({"status": "error", "message": "Missing fields"}, 400)

        col = get_collection()

        if col.find_one({'username': username}):
            return resp({"status": "error", "message": "Username already exists"}, 409)

        if col.find_one({'email': email}):
            return resp({"status": "error", "message": "Email already exists"}, 409)

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        col.insert_one({'username': username, 'password': hashed_password, 'email': email})

        return resp({"status": "success", "message": "User registered successfully"}, 201)

    except Exception as e:
        print(f"Register error: {str(e)}")
        return resp({"status": "error", "message": str(e)}, 500)


# ═══════════════════════════════════════════════════════
#  ROUTE 3 — Login
# ═══════════════════════════════════════════════════════
@app.route('/login', methods=['POST'])
def login():
    try:
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

        col = get_collection()
        user = col.find_one({'username': username})

        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            return resp({"status": "success", "username": user['username']})
        else:
            return resp({"status": "error", "message": "Invalid username or password"}, 401)

    except Exception as e:
        print(f"Login error: {str(e)}")
        return resp({"status": "error", "message": str(e)}, 500)


# ═══════════════════════════════════════════════════════
#  ROUTE 4 — Predict (CSV upload)
#  MEMORY OPTIMIZED — only 2 models, small sample
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
            return resp({"status": "error", "message": "CSV must have a 'Class' column"}, 400)

        total_transactions   = len(data)
        fraudulent_count     = int((data['Class'] == 1).sum())
        non_fraudulent_count = int((data['Class'] == 0).sum())

        print(f"Dataset: {total_transactions} rows, {fraudulent_count} fraud")

        # Very small sample to save memory
        fraud_data  = data[data['Class'] == 1]
        normal_data = data[data['Class'] == 0].sample(
                        n=min(500, len(data[data['Class'] == 0])),
                        random_state=42)
        sample = pd.concat([fraud_data, normal_data]).sample(frac=1, random_state=42)

        del data
        gc.collect()

        X = sample.drop(columns=["Class"])
        y = sample["Class"]
        del sample
        gc.collect()

        print(f"Training on {len(X)} rows...")

        # Isolation Forest
        print("Training Isolation Forest...")
        iso = IsolationForest(n_estimators=50, random_state=42, contamination=0.05)
        iso.fit(X)
        iso_preds  = [0 if p == 1 else 1 for p in iso.predict(X)]
        iso_acc    = round(accuracy_score(y, iso_preds) * 100, 2)
        iso_report = classification_report(y, iso_preds, output_dict=True)
        del iso, iso_preds
        gc.collect()
        print(f"Isolation Forest: {iso_acc}%")

        # Logistic Regression
        print("Training Logistic Regression...")
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        del X
        gc.collect()

        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X_scaled, y)
        lr_preds  = lr.predict(X_scaled)
        lr_acc    = round(accuracy_score(y, lr_preds) * 100, 2)
        lr_report = classification_report(y, lr_preds, output_dict=True)
        del lr, scaler, X_scaled, lr_preds, y
        gc.collect()
        print(f"Logistic Regression: {lr_acc}%")

        print("Done!")

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
                    "accuracy_percent": 0,
                    "note": "SVM disabled on free plan to save memory",
                    "precision": 0,
                    "recall":    0,
                    "f1_score":  0
                }
            }
        })

    except Exception as e:
        print(f"Predict error: {str(e)}")
        return resp({"status": "error", "message": str(e)}, 500)


# ═══════════════════════════════════════════════════════
#  ROUTE 5 — Logout
# ═══════════════════════════════════════════════════════
@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return resp({"status": "success", "message": "Logged out"})


# ═══════════════════════════════════════════════════════
#  ROUTE 6 — Predict Single Transaction
#  MEMORY OPTIMIZED — only 2 models
# ═══════════════════════════════════════════════════════
@app.route('/predict_single', methods=['POST'])
def predict_single():
    try:
        data = request.get_json()
        if not data:
            return resp({"status": "error", "message": "No data received"}, 400)

        sample_path = os.path.join('data', 'creditcard.csv')
        if not os.path.exists(sample_path):
            return resp({"status": "error", "message": "Training data not found on server."}, 400)

        print("Loading dataset...")
        train_df = pd.read_csv(sample_path)

        fraud_rows  = train_df[train_df['Class'] == 1]
        normal_rows = train_df[train_df['Class'] == 0].sample(
                        n=len(fraud_rows), random_state=42)
        sample = pd.concat([fraud_rows, normal_rows]).sample(frac=1, random_state=42)
        del train_df
        gc.collect()

        X_train = sample.drop(columns=['Class'])
        y_train = sample['Class']
        del sample
        gc.collect()

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
        results = {}

        # Isolation Forest
        print("Running Isolation Forest...")
        iso = IsolationForest(n_estimators=50, contamination=0.5, random_state=42)
        iso.fit(X_train)
        iso_pred  = iso.predict(X_input)[0]
        iso_score = iso.decision_function(X_input)[0]
        iso_label = "FRAUD" if iso_pred == -1 else "SAFE"
        iso_conf  = min(99.9, max(50.0, round(float(abs(iso_score) * 100), 1)))
        results['isolation_forest'] = {"prediction": iso_label, "confidence": iso_conf}
        del iso
        gc.collect()

        # Logistic Regression
        print("Running Logistic Regression...")
        scaler         = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_input_scaled = scaler.transform(X_input)
        del X_train
        gc.collect()

        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X_train_scaled, y_train)
        lr_pred  = lr.predict(X_input_scaled)[0]
        lr_proba = lr.predict_proba(X_input_scaled)[0]
        lr_label = "FRAUD" if lr_pred == 1 else "SAFE"
        lr_conf  = round(float(max(lr_proba)) * 100, 1)
        results['logistic_regression'] = {"prediction": lr_label, "confidence": lr_conf}
        del lr, scaler, X_train_scaled, X_input_scaled
        gc.collect()

        # SVM disabled
        results['svm'] = {"prediction": "N/A", "confidence": 0}

        fraud_votes = sum(1 for k, m in results.items()
                         if k != 'svm' and m['prediction'] == 'FRAUD')
        overall = "FRAUD" if fraud_votes >= 1 else "SAFE"

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