import pandas as pd
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, session, flash
from pymongo import MongoClient
import bcrypt
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from werkzeug.utils import secure_filename

app = Flask(__name__)

# for security perpose
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
        
# Define the directory where uploaded files will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
# Database
client = MongoClient(MONGO_URI)
db = client['user_database']
collection = db['users']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register')
def register_page():
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        if collection.find_one({'username': username}):
            session['error'] = "Username already exists!"
            return redirect(url_for('register_page'))
        elif collection.find_one({'email': email}):
            session['error'] = "Email already exists!"
            return redirect(url_for('register_page'))

        user_data = {
            'username': username,
            'password': hashed_password,
            'email': email
        }
        collection.insert_one(user_data)

        return redirect(url_for('login_page'))

    return render_template("login.html")

@app.route('/login')
def login_page():
    return render_template("login.html")

@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = collection.find_one({'username': username})

        if user and 'password' in user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            session['username'] = user['username']
            return redirect(url_for('dashboard', username=user['username']))
        else:
            session['error'] = "Invalid username or password"
            return redirect(url_for('login_page'))

# for security
@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        username = session['username']
        return render_template("dashboard.html", username=username)
    else:
        return redirect(url_for('login_page'))
    
@app.route('/home')
def home():
    if 'username' in session:
        username = session['username']
        return render_template("home.html", username=username)
    else:
        return redirect(url_for('login_page'))
    
@app.route('/admin')
def admin_page():
    if 'username' in session:
        username = session['username']
        return render_template("admin.html", username=username)
    else:
        return redirect(url_for('login_page'))
    

@app.route('/profile')
def profile_page():
    if 'username' in session:
        username = session['username']
        # Fetch user data from the database
        user_data = collection.find_one({'username': username})
        if user_data:
            # Pass user data to the template
            return render_template("profile.html", username=username, user_data=user_data)
        else:
            return "User not found in database"
    else:
        return redirect(url_for('login_page'))


# Add functionality to update user profile data in the database
@app.route('/profile/update', methods=['POST'])
def update_profile():
    if 'username' in session:
        username = session['username']
        # Fetch user data from the form
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        organization_name = request.form['organization_name']
        location = request.form['location']
        phone_number = request.form['phone_number']
        birthday = request.form['birthday']
        
        # Update user data in the database
        collection.update_one({'username': username}, {'$set': {
            'first_name': first_name,
            'last_name': last_name,
            'organization_name': organization_name,
            'location': location,
            'phone_number': phone_number,
            'birthday': birthday
        }})
        
        # Redirect to profile page
        return redirect(url_for('profile_page'))
    else:
        return redirect(url_for('login_page'))


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    # Read CSV in chunks to handle large files
    try:
        # Limit the number of rows to process
        data = pd.read_csv(file, nrows=10000)  # Adjust this number based on your server capacity
        
        # Basic data validation
        required_columns = ["Class"]  # Add all required columns
        if not all(col in data.columns for col in required_columns):
            return "Invalid file format: missing required columns"

        # Statistical analysis
        statistical_analysis = data.describe()
        
        # Count fraud cases
        fraudulent_count = (data['Class'] == 1).sum()
        non_fraudulent_count = (data['Class'] == 0).sum()

        # Prepare data
        X = data.drop(columns=["Class"])
        y = data["Class"]

        # Use smaller samples for training if dataset is large
        if len(X) > 5000:  # Adjust this threshold based on your needs
            from sklearn.model_selection import train_test_split
            X, _, y, _ = train_test_split(X, y, train_size=5000, random_state=42)

        # Configure models for faster processing
        iso_forest = IsolationForest(n_estimators=50, max_samples=1000, n_jobs=-1)
        svm_model = SVC(kernel='linear', max_iter=1000)
        logistic_model = LogisticRegression(max_iter=1000, n_jobs=-1)

        # Train and predict
        with parallel_backend('threading', n_jobs=2):
            # Isolation Forest
            iso_forest.fit(X)
            iso_forest_predictions = iso_forest.predict(X)
            iso_forest_accuracy = accuracy_score(y, [-1 if pred == -1 else 0 for pred in iso_forest_predictions])
            iso_forest_error = 1 - iso_forest_accuracy
            iso_forest_classification_report = classification_report(y, [-1 if pred == -1 else 0 for pred in iso_forest_predictions])

            # SVM
            svm_model.fit(X, y)
            svm_predictions = svm_model.predict(X)
            svm_accuracy = accuracy_score(y, svm_predictions)
            svm_error = 1 - svm_accuracy
            svm_classification_report = classification_report(y, svm_predictions)

            # Logistic Regression
            logistic_model.fit(X, y)
            logistic_predictions = logistic_model.predict(X)
            logistic_accuracy = accuracy_score(y, logistic_predictions)
            logistic_error = 1 - logistic_accuracy
            logistic_classification_report = classification_report(y, logistic_predictions)

        return render_template('admin.html', 
                            username=session['username'],
                            statistical_analysis=statistical_analysis,
                            fraudulent_count=fraudulent_count,
                            non_fraudulent_count=non_fraudulent_count,
                            iso_forest_accuracy=iso_forest_accuracy,
                            iso_forest_error=iso_forest_error,
                            iso_forest_classification_report=iso_forest_classification_report,
                            svm_accuracy=svm_accuracy,
                            svm_error=svm_error,
                            svm_classification_report=svm_classification_report,
                            logistic_accuracy=logistic_accuracy,
                            logistic_error=logistic_error,
                            logistic_classification_report=logistic_classification_report)
                            
    except Exception as e:
        return f"Error processing file: {str(e)}"

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)