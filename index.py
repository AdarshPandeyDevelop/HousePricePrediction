# House Price Prediction Project
# Python | Pandas | NumPy | scikit-learn | XGBoost

# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Load Data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Exploratory Data Analysis (EDA)
# Histogram of SalePrice
plt.figure(figsize=(8, 6))
sns.histplot(train['SalePrice'], kde=True)
plt.title("Distribution of House Prices")
plt.savefig("saleprice_distribution.png")  # Save plot
plt.close()

# Correlation heatmap
corr = train.corr(numeric_only=True)
plt.figure(figsize=(12, 10))
sns.heatmap(corr[['SalePrice']].sort_values(
    by='SalePrice', ascending=False), annot=True)
plt.title("Feature Correlation with SalePrice")
plt.savefig("correlation_heatmap.png")
plt.close()

# Data Cleaning

# Remove outliers
train = train[train['GrLivArea'] < 4500]

# Fill numeric missing values with median
train.fillna(train.median(numeric_only=True), inplace=True)

# Feature Engineering

# Create new features
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
train['HouseAge'] = train['YrSold'] - train['YearBuilt']

# Encode categorical variables
for col in train.select_dtypes('object'):
    lbl = LabelEncoder()
    train[col] = lbl.fit_transform(train[col].astype(str))

# Split Data
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train Models

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
}

print("\n--- Model Evaluation on Validation Set ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    print(f"{name} R² score: {r2_score(y_valid, preds):.4f}")


# Cross-Validation

print("\n--- Cross-Validation Scores (5-Fold) ---")
for name, model in models.items():
    scores = cross_val_score(model, X, y, scoring='r2', cv=5)
    print(f"{name} Mean CV R²: {scores.mean():.4f}")


# Prediction Visualization (Optional)

best_model = models["XGBoost"]  # Assuming XGBoost is best
preds = best_model.predict(X_valid)

plt.figure(figsize=(8, 6))
plt.scatter(y_valid, preds, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.savefig("actual_vs_predicted.png")
plt.close()

print("\nProject complete! Visualizations saved as PNG files.")
