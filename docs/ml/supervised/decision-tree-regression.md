---
title: Decision Tree Regression
sidebar_position: 4
description: Understanding Decision Tree algorithms for regression tasks
---

# Decision Tree Regression

Decision Trees for regression are **supervised learning algorithms** used to predict continuous target variables. Unlike classification trees that predict categories, regression trees predict numerical values by splitting the feature space and approximating the target with the mean value in each region.

## 1. How Regression Trees Work

Decision trees use a **hierarchical, tree-like structure** where:
- **Root Node**: Starting point, represents entire dataset
- **Decision Nodes**: Points where data is split based on feature values
- **Leaf Nodes**: Terminal nodes that provide the numerical output (typically the mean of target values in that region)

The algorithm works by:
1. **Selecting the best feature** to split the data at each node
2. **Creating child nodes** based on the split
3. **Repeating recursively** until stopping criteria are met

### Splitting Criteria:
- **Mean Squared Error (MSE)**: The most common criterion, minimizes the average of squared differences between target and predicted values
- **Mean Absolute Error (MAE)**: Minimizes the average of absolute differences
- **Friedman's MSE**: A modified version of MSE used in some implementations

### Stopping Criteria:
- Maximum depth reached
- Minimum samples required for a split
- Minimum samples required at a leaf node
- Maximum leaf nodes reached

---

## 2. Example Use Case: House Price Prediction

### Scenario
A real estate company wants to predict house prices based on property characteristics.

### Dataset Sample

| Size (sqft) | Bedrooms | Age (years) | Location Rating | Price ($) |
|-------------|----------|-------------|----------------|-----------|
| 1200        | 2        | 15          | 7              | 250,000   |
| 2400        | 4        | 5           | 9              | 550,000   |
| 1500        | 3        | 10          | 6              | 320,000   |
| 1800        | 3        | 20          | 8              | 380,000   |
| 3000        | 5        | 2           | 10             | 750,000   |

### Decision Tree Approach

The algorithm will:
1. Select the most informative feature for the first split (e.g., Size)
2. Split the data based on this feature
3. Continue splitting recursively on each subset
4. Form a tree structure that can predict house prices for new properties

For example, the resulting tree might look like:
```
Size ≤ 1500 sqft?
├── Yes → Location Rating ≤ 7?
│   ├── Yes → $230,000
│   └── No → $290,000
└── No → Bedrooms ≤ 3?
    ├── Yes → Age ≤ 10?
    │   ├── Yes → $400,000
    │   └── No → $350,000
    └── No → $650,000
```

For a new property with:
- Size: 1800 sqft
- Bedrooms: 3
- Age: 7 years
- Location Rating: 8

We would trace the path: Size > 1500 sqft → Bedrooms ≤ 3 → Age ≤ 10 → $400,000

---

## 3. Advantages of Decision Trees for Regression

✅ Simple to understand and visualize decision boundaries

✅ Requires little data preprocessing (no normalization needed)

✅ Can handle both numerical and categorical data

✅ Implicitly performs feature selection

✅ Non-parametric (no assumptions about data distribution)

✅ Handles non-linear relationships well

---

## 4. Limitations

❌ Prone to overfitting, especially with deep trees

❌ Poor at extrapolation (predictions outside the range of training data)

❌ Can be unstable (small variations in data can produce very different trees)

❌ May struggle with smooth, continuous functions

Techniques like **pruning**, **setting maximum depth**, and **ensemble methods** (Random Forests) can address some of these limitations.

---

## 5. Real-World Applications

🏠 **Real Estate Valuation** – Predicting property prices

📈 **Financial Forecasting** – Predicting stock prices or economic indicators

⛅ **Weather Prediction** – Forecasting temperature, rainfall, etc.

⚡ **Energy Consumption** – Estimating household or industrial energy usage

🚗 **Vehicle Pricing** – Determining used car values

---

## 6. Implementation Example

```python
# Install Dependencies
# pip install numpy pandas scikit-learn matplotlib seaborn 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_graphviz
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import math

# Sample house price dataset
data = {
    'Size': [1200, 2400, 1500, 1800, 3000, 1650, 2100, 2700, 1400, 1950, 
              2200, 1300, 2800, 3200, 1750, 2500, 1600, 2300, 3500, 2000],
    'Bedrooms': [2, 4, 3, 3, 5, 3, 4, 4, 2, 3, 
                 4, 2, 5, 5, 3, 4, 3, 4, 6, 3],
    'Age': [15, 5, 10, 20, 2, 12, 8, 3, 25, 15, 
            10, 30, 4, 1, 18, 7, 9, 6, 2, 11],
    'LocationRating': [7, 9, 6, 8, 10, 7, 8, 9, 5, 7, 
                       8, 6, 9, 10, 7, 8, 6, 8, 10, 7],
    'Price': [250000, 550000, 320000, 380000, 750000, 340000, 470000, 620000, 230000, 400000,
              480000, 210000, 670000, 820000, 360000, 580000, 330000, 510000, 900000, 420000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the first few rows
print(df.head())

# Basic data exploration
print("\nData Summary:")
print(df.describe())

# Visualize the data
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.scatter(df['Size'], df['Price'])
plt.title('Size vs. Price')
plt.xlabel('Size (sqft)')
plt.ylabel('Price ($)')

plt.subplot(2, 2, 2)
plt.scatter(df['Bedrooms'], df['Price'])
plt.title('Bedrooms vs. Price')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price ($)')

plt.subplot(2, 2, 3)
plt.scatter(df['Age'], df['Price'])
plt.title('Age vs. Price')
plt.xlabel('Age (years)')
plt.ylabel('Price ($)')

plt.subplot(2, 2, 4)
plt.scatter(df['LocationRating'], df['Price'])
plt.title('Location Rating vs. Price')
plt.xlabel('Location Rating')
plt.ylabel('Price ($)')

plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Prepare data for the model
X = df[['Size', 'Bedrooms', 'Age', 'LocationRating']]
y = df['Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Cross-validation to find optimal max_depth
depth_range = range(1, 10)
mse_scores = []

for depth in depth_range:
    dt_regressor = DecisionTreeRegressor(max_depth=depth, random_state=42)
    # Use negative MSE as scoring since cross_val_score maximizes the score
    scores = cross_val_score(dt_regressor, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mse_scores.append(-scores.mean())  # Convert back to positive MSE

# Plot cross-validation results
plt.figure(figsize=(10, 6))
plt.plot(depth_range, mse_scores, marker='o')
plt.xlabel('Maximum Depth')
plt.ylabel('Mean Squared Error')
plt.title('Finding Optimal Tree Depth')
plt.grid(True)
plt.show()

# Convert MSE to RMSE for easier interpretation
rmse_scores = [math.sqrt(mse) for mse in mse_scores]
plt.figure(figsize=(10, 6))
plt.plot(depth_range, rmse_scores, marker='o')
plt.xlabel('Maximum Depth')
plt.ylabel('Root Mean Squared Error')
plt.title('Finding Optimal Tree Depth (RMSE)')
plt.grid(True)
plt.show()

# Get the best depth
best_depth = depth_range[np.argmin(mse_scores)]
print(f"\nOptimal tree depth: {best_depth}")

# Train the final model with the best depth
dt_regressor = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
dt_regressor.fit(X_train, y_train)

# Make predictions
y_pred = dt_regressor.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')

# Add perfect prediction line
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.show()

# Visualize residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted Price')
plt.ylabel('Residual')
plt.title('Residual Plot')
plt.show()

# Visualize the decision tree
plt.figure(figsize=(20, 10))
feature_names = ['Size (sqft)', 'Bedrooms', 'Age (years)', 'Location Rating']
plot_tree(dt_regressor, filled=True, feature_names=feature_names, rounded=True)
plt.title(f"Decision Tree Regressor (Max Depth = {best_depth})")
plt.show()

# Feature importance
importances = dt_regressor.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), np.array(feature_names)[indices])
plt.tight_layout()
plt.show()

print("\nFeature Importances:")
for i, feature in enumerate(np.array(feature_names)[indices]):
    print(f"{feature}: {importances[indices[i]]:.4f}")

# Predict price for a new house
new_house = np.array([[1800, 3, 7, 8]])
prediction = dt_regressor.predict(new_house)

print("\nNew House:")
print(f"Size: 1800 sqft")
print(f"Bedrooms: 3")
print(f"Age: 7 years")
print(f"Location Rating: 8")
print(f"\nPredicted Price: ${prediction[0]:,.2f}")

# Find the path through the decision tree for the new house
def get_decision_path(tree, X, feature_names):
    node_indicator = tree.decision_path(X)
    leaf_id = tree.apply(X)
    
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    value = tree.tree_.value
    
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    
    rules = []
    for node_id in node_index:
        # Continue to the next node if not a leaf
        if leaf_id[0] != node_id:
            # If the feature value is below the threshold, follow the left path
            if X[0, feature[node_id]] <= threshold[node_id]:
                threshold_sign = "<="
            else:
                threshold_sign = ">"
                
            rules.append(f"{feature_names[feature[node_id]]} {threshold_sign} {threshold[node_id]}")
        else:
            # If it's a leaf node, get the prediction value
            rules.append(f"Predicted Price: ${value[node_id][0][0]:,.2f}")
    
    return rules

decision_path = get_decision_path(dt_regressor, new_house, feature_names)
print("\nDecision Path:")
for i, rule in enumerate(decision_path):
    print(f"{i+1}. {rule}")
```

## Summary

✅ Built a Decision Tree regression model for house price prediction

✅ Used cross-validation to find the optimal tree depth

✅ Visualized the decision tree for interpretability

✅ Analyzed feature importance to understand key factors

✅ Demonstrated making and explaining predictions for new houses with exact values

✅ Evaluated model using RMSE, MAE, and R² metrics

## Comparing Regression Trees to Classification Trees

| Aspect | Classification Trees | Regression Trees |
|--------|---------------------|------------------|
| **Target Variable** | Categorical (classes) | Numerical (continuous) |
| **Leaf Nodes** | Store class probabilities | Store mean target values |
| **Splitting Criteria** | Gini impurity, Entropy | MSE, MAE |
| **Prediction** | Class label or probabilities | Numerical value |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1 | RMSE, MAE, R² |
| **Visualization** | Classes shown with colors | Values shown in leaves |