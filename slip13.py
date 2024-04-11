'''Download nursery dataset from UCI. Build a linear regression model by identifying independent
and target variable. Split the variables into training and testing sets and print them. Build a simple linear
regression model for predicting purchases.'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data"
column_names = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health", "class"]
df = pd.read_csv(url, header=None, names=column_names)

# Identify independent and target variables
X = df.drop("class", axis=1)
y = df["class"]

# Convert categorical variables into dummy variables
X_encoded = pd.get_dummies(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# Print the training and testing sets
print("Training set:")
print(X_train.head())
print("\nTesting set:")
print(X_test.head())

# Build a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
