'''Create ‘User’ Data set having 5 columns namely: User ID, Gender, Age, Estimated Salary and
Purchased. Build a logistic regression model that can predict whether on the given parameter a person
will buy a car or not.'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Creating the dataset
data = {'User ID': [1, 2, 3, 4, 5],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Age': [25, 30, 35, 40, 45],
        'Estimated Salary': [50000, 60000, 70000, 80000, 90000],
        'Purchased': [0, 1, 0, 1, 0]}  # 0 represents not purchased, 1 represents purchased

df = pd.DataFrame(data)

# Convert categorical variables into dummy/indicator variables
df = pd.get_dummies(df, columns=['Gender'])

# Identify independent and target variables
X = df[['Age', 'Estimated Salary', 'Gender_Female', 'Gender_Male']]
y = df['Purchased']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)
