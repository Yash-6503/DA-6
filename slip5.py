'''Use the iris dataset. Write a Python program to view some basic statistical details like percentile,
mean, std etc. of the species of 'Iris-setosa', 'Iris-versicolor' and 'Iris-virginica'. Apply logistic regression
on the dataset to identify different species (setosa, versicolor, verginica) of Iris flowers given just 4
features: sepal and petal lengths and widths.. Find the accuracy of the model.'''

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the iris dataset
df = pd.read_csv("./Datasets/Iris.csv")

# View basic statistical details of each species
setosa_stats = df[['species'] == 'Iris-setosa'].describe()
versicolor_stats = df[['species'] == 'Iris-versicolor'].describe()
virginica_stats = df[['species'] == 'Iris-virginica'].describe()

print("Statistical details for Iris-setosa:")
print(setosa_stats)
print("\nStatistical details for Iris-versicolor:")
print(versicolor_stats)
print("\nStatistical details for Iris-virginica:")
print(virginica_stats)

# Apply logistic regression
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Predict species
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy of the logistic regression model:", accuracy)
