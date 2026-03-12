#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 01:31:31 2026

@author: tayfunburakaksoy
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1) Load dataset (historial data)
df = pd.read_csv("Historical_Customer_Data.csv", sep=";")

# 2) Check first rows
print(df.head())
print(df.info())

# 3) Define target variable
y = df["churned"]

# 4) Define independent variables
# We drop customer_id because it is only an identifier
X = df.drop(columns=["customer_id", "churned"])

# 5) Convert categorical variables into dummy variables
X = pd.get_dummies(X, drop_first=True)

# 6) Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 7) Build logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8) Predict on test set
y_pred = model.predict(X_test)

# 9) Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 10) Load current customer data
current_df = pd.read_csv("Current_Customer_Data.csv", sep=";")

# Convert commas on csv file to decimal points (to fix errors)
decimal_cols = ["watch_hours", "monthly_fee", "avg_watch_time_per_day"]

for col in decimal_cols:
    current_df[col] = current_df[col].astype(str).str.replace(",", ".", regex=False)
    current_df[col] = pd.to_numeric(current_df[col], errors="coerce")

# Check data types
print(current_df.dtypes)

# Keep customer_id separately
customer_ids = current_df["customer_id"]

# Remove customer_id from predictors
current_X = current_df.drop(columns=["customer_id"])

# Convert categorical variables to dummy variables
current_X = pd.get_dummies(current_X, drop_first=True)


# 11) Match columns with training data
current_X = current_X.reindex(columns=X.columns, fill_value=0)


# 12) Predict churn probabilities
churn_prob = model.predict_proba(current_X)[:, 1]

# Convert probabilities into 0/1 classes
churn_pred = model.predict(current_X)

# Create results table

results = pd.DataFrame({
    "customer_id": customer_ids,
    "predicted_probability": churn_prob,
    "predicted_churn_class": churn_pred
})

# Show first rows
print(results.head(10))

# Save results
results.to_csv("Predictions_Output.csv", index=False)
















