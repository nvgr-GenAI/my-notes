---
title: Linear Regression
sidebar_position: 2
description: Understanding Linear Regression algorithm, types, usage, and implementation
---

# Linear Regression

Linear Regression is a **supervised learning algorithm** used for predicting a continuous dependent variable (Y)based on one or more independent variables (X). It assumes a linear relationship between the input and output variables.

## 1. Types of Linear Regression

### A. Simple Linear Regression

It involves one independent variable and one dependent variable, modeled as:

Y = mX + b

Where:

- Y = Dependent variable (target)
- X = Independent variable (feature)
- m = Slope of the line (coefficient)
- b = Intercept (constant)

### B. Multiple Linear Regression

It involves two or more independent variables and is represented as:

Y = b₀ + b₁X₁ + b₂X₂ + ... + bₙXₙ

Where:

- X₁, X₂, ..., Xₙ are multiple independent variables
- b₀ is the intercept
- b₁, b₂, ..., bₙ are coefficients

---

## 2. How Linear Regression Works

Linear regression works by finding the best-fit line that minimizes the difference between predicted and actual values. The optimization technique used is **Ordinary Least Squares (OLS)**, which minimizes the **Mean Squared Error (MSE)**:

MSE = (1/n) Σ(Yᵢ - Ŷᵢ)²

Where:

- Yᵢ = Actual values
- Ŷᵢ = Predicted values

The coefficients are found using:

B = (XᵀX)⁻¹XᵀY

---

## 3. Example Use Case: Predicting House Prices

### Scenario

A real estate company wants to predict house prices based on features like area (sq ft), number of bedrooms, and distance from the city center.

### Dataset Sample

| Area (sq ft) | Bedrooms | Distance to City (km) | Price ($) |
| --- | --- | --- | --- |
| 1500 | 3 | 5 | 300,000 |
| 1800 | 4 | 7 | 350,000 |
| 1200 | 2 | 3 | 200,000 |

### Applying Multiple Linear Regression

The model will learn the relationship:

Price = b₀ + b₁(Area) + b₂(Bedrooms) + b₃(Distance)

If the trained model gives:

Price = 50,000 + 150(Area) + 20,000(Bedrooms) - 5,000(Distance)

For a new house with:

- 1600 sq ft area
- 3 bedrooms
- 4 km from city

Price = 50,000 + 150(1600) + 20,000(3) - 5,000(4) = 320,000

So, the predicted price is **$320,000**.

---

## 4. Advantages of Linear Regression

✅ Easy to interpret and implement

✅ Works well when there's a linear relationship

✅ Computationally efficient

---

## 5. Limitations

❌ Assumes linearity between variables

❌ Sensitive to outliers

❌ Doesn't handle complex relationships (non-linearity)

For complex relationships, **Polynomial Regression** or **Neural Networks** might be better alternatives.

---

## 6. Real-World Applications

📊 **Stock Market Prediction** – Predicting stock prices based on historical trends.

🏡 **Real Estate Pricing** – Estimating property values based on features.

🚗 **Fuel Consumption** – Predicting fuel efficiency based on engine specs.

📈 **Sales Forecasting** – Predicting future sales based on past data.

## 7. Implementation Example

```python
# Install Dependencies
# pip install numpy pandas scikit-learn matplotlib seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Sample housing dataset
data = {
    "Area (sq ft)": [1500, 1800, 1200, 2500, 1600, 2000, 2200, 1700, 1400, 1900],
    "Bedrooms": [3, 4, 2, 5, 3, 4, 4, 3, 2, 4],
    "Distance to City (km)": [5, 7, 3, 10, 4, 6, 8, 5, 3, 6],
    "Price ($)": [300000, 350000, 200000, 500000, 320000, 400000, 450000, 330000, 250000, 370000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the first few rows
print(df.head())

#Visualize the Data
sns.pairplot(df, diag_kind="kde")
plt.show()

#Prepare the Data for Training
# Define features (X) and target (y)
X = df[["Area (sq ft)", "Bedrooms", "Distance to City (km)"]]
y = df["Price ($)"]

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Print learned coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

# Predict on test data
y_pred = model.predict(X_test)

# Compare actual vs predicted prices
df_results = pd.DataFrame({"Actual Price": y_test, "Predicted Price": y_pred})
print(df_results)

#Evaluate the Model
# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared Score (R²): {r2}")

#Make a Custom Prediction
# Predict price for a new house
new_house = np.array([[1800, 3, 5]])  # 1800 sq ft, 3 bedrooms, 5 km from city
predicted_price = model.predict(new_house)

print(f"Predicted Price for house (1800 sq ft, 3 bedrooms, 5 km away): ${predicted_price[0]:,.2f}")

# Visualize Actual vs Predicted Prices
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title("Actual vs Predicted House Prices")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="dashed")  # Perfect fit line
plt.show()
```

## Summary

✅ Built a Linear Regression model to predict house prices

✅ Trained the model on sample data

✅ Made predictions on unseen data

✅ Evaluated the model using MAE, RMSE, and R²

✅ Plotted actual vs predicted prices