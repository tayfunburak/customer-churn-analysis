#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:35:20 2026

@author: tayfunburakaksoy
"""


import pandas as pd
import matplotlib.pyplot as plt

# 1) Load prediction output file
df = pd.read_csv("Predictions_Output.csv")

# 2) General information
print("First 5 rows:")
print(df.head())

print("\nColumn types:")
print(df.dtypes)

# 3) Descriptive statistics for predicted probabilities
print("\nDescriptive statistics of predicted probabilities:")
print(df["predicted_probability"].describe())

# 4) Count predicted classes
print("\nPredicted churn class counts:")
print(df["predicted_churn_class"].value_counts())

print("\nPredicted churn class percentages:")
print(df["predicted_churn_class"].value_counts(normalize=True) * 100)

# 5) Probability groups
bins = [0, 0.01, 0.05, 0.20, 0.50, 0.80, 1.00]
labels = ["0-0.01", "0.01-0.05", "0.05-0.20", "0.20-0.50", "0.50-0.80", "0.80-1.00"]

df["prob_group"] = pd.cut(df["predicted_probability"], bins=bins, labels=labels, include_lowest=True)

print("\nCustomers in each probability group:")
print(df["prob_group"].value_counts().sort_index())


# GRAPHS

# 1) Histogram
plt.figure(figsize=(8,5))
plt.hist(df["predicted_probability"], bins=30, edgecolor="black")
plt.xlabel("Predicted Probability")
plt.ylabel("Number of Customers")
plt.title("Distribution of Predicted Churn Probabilities")
plt.show()

# 2) Boxplot
plt.figure(figsize=(8,4))
plt.boxplot(df["predicted_probability"], vert=False)
plt.xlabel("Predicted Probability")
plt.title("Boxplot of Predicted Churn Probabilities")
plt.show()

# 3) Bar chart for class counts
class_counts = df["predicted_churn_class"].value_counts().sort_index()

plt.figure(figsize=(6,4))
plt.bar(class_counts.index.astype(str), class_counts.values, edgecolor="black")
plt.xlabel("Predicted Churn Class")
plt.ylabel("Number of Customers")
plt.title("Predicted Churn Class Counts")
plt.show()

# 4) Bar chart for probability groups
group_counts = df["prob_group"].value_counts().sort_index()

plt.figure(figsize=(8,5))
plt.bar(group_counts.index.astype(str), group_counts.values, edgecolor="black")
plt.xlabel("Probability Group")
plt.ylabel("Number of Customers")
plt.title("Customers by Probability Range")
plt.xticks(rotation=30)
plt.show()


















