'''Create ‘Salary’ Data set . Build a linear regression model by identifying independent and target
variable. Split the variables into training and testing sets and print them. Build a simple linear regression
model for predicting purchases. '''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Creating the dataset
data = {'Age': [25, 30, 35, 40, 45, 50, 55, 60],
        'Salary': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
        'Purchases': [10, 12, 15, 18, 20, 22, 25, 28]}

df = pd.DataFrame(data)

# Identify independent and target variables
X = df[['Age', 'Salary']]
y = df['Purchases']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training set:")
print(X_train)
print(y_train)
print("")

print("Testing set:")
print(X_test)
print(y_test)
print("")

# Build a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
