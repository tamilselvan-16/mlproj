# mlproj




# Project Title:
  Customer ML-Analysis
# Aim:
  Using customer retail dataset implement various models and compare them.

# Project objective:

The objective of this project is to build and compare multiple Machine Learning models using a customer dataset. In this we will learn preprocessing, visualization, model training, prediction, evaluation, and comparison of algorithms.

# Dataset Used:

• Quantity 

• UnitPrice

• Country 

# Machine Learning Models Used :
• Logistic Regression 

• Decision Tree Classifier

• K-Nearest Neighbors (KNN)

# Project Workflow:

1. Load customer dataset using Pandas

2. Handle missing values

3. Encode categorical columns using LabelEncoder

4. Visualize customer data using Matplotlib

5. Split data into training and testing sets

6. Train Logistic Regression model

7. Train Decision Tree model

8. Train KNN model

9. Evaluate models using Accuracy and Confusion Matrix

10. Compare model performances using graphs

# Program:

```
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# =========================================
# LOAD DATASET
# =========================================

data = pd.read_csv("customer_data.csv")

# Display first 5 rows
print("First 5 Rows of Dataset")
print(data.head())

# =========================================
# HANDLE MISSING VALUES
# =========================================

data = data.dropna()

# =========================================
# KEEP ONLY REQUIRED COLUMNS
# =========================================

data = data[['Quantity', 'UnitPrice', 'Country']]

# =========================================
# ENCODE COUNTRY COLUMN
# =========================================

le = LabelEncoder()

data['Country'] = le.fit_transform(data['Country'])

# =========================================
# FEATURES AND TARGET
# =========================================

X = data[['Quantity', 'UnitPrice']]

y = data['Country']

# =========================================
# FEATURE SCALING
# =========================================

scaler = StandardScaler()

X = scaler.fit_transform(X)

# =========================================
# DATA VISUALIZATION
# =========================================

plt.figure(figsize=(6,4))

plt.hist(data['Quantity'])

plt.title("Customer Quantity Distribution")

plt.xlabel("Quantity")

plt.ylabel("Count")

plt.show()

# =========================================
# SPLIT DATA
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================
# LOGISTIC REGRESSION
# =========================================

print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")

lr_model = LogisticRegression(max_iter=2000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)

print("\nAccuracy:")
print(lr_accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_pred))

# Graphical Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, lr_pred)

plt.title("Logistic Regression Confusion Matrix")

plt.show()

# =========================================
# DECISION TREE
# =========================================

print("\n==============================")
print("DECISION TREE")
print("==============================")

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)

print("\nAccuracy:")
print(dt_accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, dt_pred))

# Graphical Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, dt_pred)

plt.title("Decision Tree Confusion Matrix")

plt.show()

# =========================================
# KNN MODEL
# =========================================

print("\n==============================")
print("KNN MODEL")
print("==============================")

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

knn_pred = knn_model.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_pred)

print("\nAccuracy:")
print(knn_accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, knn_pred))

# Graphical Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, knn_pred)

plt.title("KNN Confusion Matrix")

plt.show()

# =========================================
# MODEL COMPARISON GRAPH
# =========================================

models = [
    'Logistic Regression',
    'Decision Tree',
    'KNN'
]

accuracies = [
    lr_accuracy,
    dt_accuracy,
    knn_accuracy
]

plt.figure(figsize=(8,5))

plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")

plt.xlabel("Models")

plt.ylabel("Accuracy")

plt.show()

# =========================================
# FINAL ACCURACY OUTPUT
# =========================================

print("\n===================================")
print("FINAL MODEL ACCURACY COMPARISON")
print("===================================")

print("Logistic Regression Accuracy :", lr_accuracy)

print("Decision Tree Accuracy       :", dt_accuracy)

print("KNN Accuracy                 :", knn_accuracy)
```

# Output:

First 5 Rows of Dataset

<img width="687" height="305" alt="image" src="https://github.com/user-attachments/assets/c4eaf779-0dc3-4f80-a858-29f5b04b0c6a" />

<img width="755" height="516" alt="image" src="https://github.com/user-attachments/assets/e1169f99-cd6b-4dfc-bc6a-110c362a6a15" />

Logistic Regression

<img width="505" height="335" alt="image" src="https://github.com/user-attachments/assets/1c73c35d-092e-4db6-a5ed-7ef1b83f82be" />

Decision Tree

<img width="515" height="331" alt="image" src="https://github.com/user-attachments/assets/f0dfcd1f-dc69-4418-9a6a-5817c6a322c6" />

KNN model

<img width="478" height="350" alt="image" src="https://github.com/user-attachments/assets/5d70b1d2-8c32-40eb-b1d6-5ccaba73ff76" />

<img width="875" height="593" alt="image" src="https://github.com/user-attachments/assets/065968af-ac3c-49be-aef3-1ee9b54b298d" />


<img width="658" height="137" alt="image" src="https://github.com/user-attachments/assets/28482abb-d262-4d48-bbab-310a890cd50c" />

# Conclusion:

Thus,the project demonstrates how different Machine Learning algorithms perform on the same dataset and sucessfully executed.


      
