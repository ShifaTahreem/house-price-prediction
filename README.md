# house-price-prediction
END-TO-END DATA SCIENCE PROJECT

COMPANY:CODECH IT SOLUTIONS

INTERN ID:CT04DZ1769

DOMAIN:DATA SCIENCE

DURATION:4 WEEKS

MENTOR:NEELA SANTOSH

Project Title: House Price Prediction Using Machine Learning with Flask Deployment

Project Description:
This project was completed as a part of my machine learning and internship learning journey. The objective was to design and implement an end-to-end house price prediction system using regression techniques. I leveraged the scikit-learn library for training the model and Flask for deployment, allowing users to interact with the model via a user-friendly web interface.

House price prediction is a common real-world use case in the domain of machine learning, especially in the real estate sector. This project demonstrates how raw data can be transformed into an intelligent system that can provide instant predictions using a machine learning model hosted through a web application.

This task also formed a part of my minor academic project submission, helping me gain practical experience in the complete machine learning workflow, from data preprocessing and model training to real-time deployment and testing.

Dataset Used:
I used the "House Prices - Advanced Regression Techniques" dataset from Kaggle. This dataset is widely used for regression-based ML tasks and contains a rich collection of house-related features.

Source: https://www.kaggle.com/c/house-prices-advanced-regression-techniques

Training Samples: ~1,460

Test Samples: ~1,450 (without target variable)

Target Variable: SalePrice (The price the house was sold for)

In this project, I focused on five numeric features which are commonly correlated with house price:

GrLivArea – Above Ground Living Area (in square feet)

TotalBsmtSF – Total Basement Area (in square feet)

GarageArea – Size of the Garage (in square feet)

FullBath – Number of Full Bathrooms

OverallQual – Overall Quality Rating (1–10)

These features were selected after exploring the data for correlation with the target variable.

Data Preprocessing:
Before training the model, the following preprocessing steps were applied:

Handpicked the most relevant numerical features

Checked for missing values and handled them (if any)

Performed exploratory data analysis to understand data distribution

Split the data into training and testing sets (80/20 ratio)

No categorical encoding or scaling was required as all selected features were numeric

The train_model.py script handles all preprocessing steps internally.

Model Selection:
I chose a Linear Regression model from scikit-learn (LinearRegression) due to its simplicity, interpretability, and effectiveness for continuous target prediction tasks. It’s a foundational algorithm that helps in understanding how machine learning models work under the hood.

Model Training:
The model was trained using the train_model.py script which performs the following tasks:

Reads the train.csv dataset

Extracts features and target column

Splits the data into training and test sets

Trains a Linear Regression model

Evaluates performance using:

R² Score (coefficient of determination)

RMSE (Root Mean Squared Error)

Saves the model as model.pkl using Python’s pickle library

Model Performance:
R² Score: ~0.75 (i.e., the model explains 75% of the variance in the target variable)

RMSE: Around ₹39,000 (i.e., average prediction error is ~₹39,000)

Web Application Using Flask:
I used Flask, a lightweight Python web framework, to create a user interface for the model. The app allows users to input values for the five features and receive an instant prediction.

app.py: Flask backend server

templates/index.html: Frontend HTML page for user input

On form submission, the inputs are processed and sent to the model

The model predicts the price and returns the result in the browser

Sample Workflow:
User opens http://127.0.0.1:5000/

Inputs:

GrLivArea = 1800

TotalBsmtSF = 1200

GarageArea = 400

FullBath = 2

OverallQual = 7

Clicks “Predict”

Output: Predicted Price: ₹215,823.56

Sample outputs:

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/dd8eec15-aefc-4d1e-b591-d06a96b4c83e" />
