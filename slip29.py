'''Build a logistic regression model for Student Score Dataset'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Import Libraries

# Step 2: Load Dataset
data = pd.read_csv("student_scores.csv")

# Step 3: Preprocess Data
# Assuming the dataset has features 'hours_studied' and 'passed_exam'
X = data[['hours_studied']]
y = data['passed_exam']

# Step 4: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build Logistic Regression Model
model = LogisticRegression()

# Step 6: Train Model
model.fit(X_train, y_train)

# Step 7: Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
