Credit Card Fraud Detection with Flask and Machine Learning
===========================================================
[![💳 LINK](https://img.shields.io/badge/💳-Visit%20Website-blue)](https://ccdf.onrender.com/)

> due to some problem in render, ML model may not work in Website but locally it works fine.

## Video Preview

[![Watch the video](https://img.youtube.com/vi/q8WpF-j_wsE/0.jpg)](https://youtu.be/q8WpF-j_wsE)

Project Overview
----------------

This project aims to detect fraudulent credit card transactions using machine learning algorithms. It includes a web application built with Flask where users can upload transaction data, which is then processed using machine learning models to detect fraud.

Installation
------------

Prerequisites:
- Python 3.x installed on your system.
- MongoDB installed locally or accessible remotely.
- Git installed to clone the repository.

Steps:
1. Clone the repository:

   git clone https://github.com/therealdope/Credit_Card_Fraud_Detection_Website.git
   
   cd Credit_Card_Fraud_Detection_Website

3. Set up Python environment:

   python -m venv env
   
   source `env/bin/activate`   #On Windows use `env\Scripts\activate`

5. Install dependencies:

   pip install -r requirements.txt

6. Download the dataset:

   - The dataset used in this project is not included in the repository due to size constraints.
   - Download the dataset from https://drive.google.com/file/d/1GNxFy8jlTZQLny81XoaYOqfQDQWNFQgh/view?usp=sharing and place it in a directory (data/) in the root of your project.

7. Set up MongoDB:

   - Install MongoDB on your local machine or use a cloud-based MongoDB service.
   - Configure MongoDB connection URI in app.py or a separate configuration file.

8. Run the application:

   python app.py or flask run

   The application should now be running locally. Access it at http://localhost:5000 in your web browser.

Project Structure
-----------------

- app.py: Flask application setup and routes and Machine learning models and preprocessing scripts..
- templates/: HTML templates for rendering frontend.
- static/: CSS stylesheets and other static files.
- data/: Directory to store the dataset (not included in repository).

Machine Learning Models
-----------------------

This project uses the following machine learning models from scikit-learn for fraud detection:

- Isolation Forest
- Support Vector Classifier (SVC)
- Logistic Regression

Libraries used include pandas, Flask, pymongo, bcrypt, and werkzeug for file uploads.

Contributing
------------

Contributions are welcome! Please fork the repository and create a pull request for any improvements or fixes.
