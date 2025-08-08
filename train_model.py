import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
data = pd.read_csv('train.csv')

# Select relevant features
features = ['GrLivArea', 'TotalBsmtSF', 'GarageArea', 'FullBath', 'OverallQual']
target = 'SalePrice'

# Drop rows with missing values
data = data[features + [target]].dropna()

# Define X and y
X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Save model
joblib.dump(model, 'model.pkl')
print("✅ Model saved as model.pkl")
