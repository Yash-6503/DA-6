'''Create ‘Position_Salaries’ Data set. Build a linear regression model by identifying independent and
target variable. Split the variables into training and testing sets. then divide the training and testing sets
into a 7:3 ratio, respectively and print them. Build a simple linear regression model.'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Creating the dataset
data = {'Position': ['Manager', 'Team Leader', 'Software Engineer', 'Intern', 'Manager', 'Team Leader', 'Software Engineer', 'Intern'],
        'Level': [1, 2, 3, 4, 5, 6, 7, 8],
        'Salary': [45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000]}

df = pd.DataFrame(data)

# Identify independent and target variables
X = df[['Level']]
y = df['Salary']

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
print("Coefficient:", model.coef_)
