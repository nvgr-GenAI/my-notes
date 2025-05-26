# Random Forest Regression

Random Forest Regression is an **ensemble learning method** that combines multiple decision trees to predict continuous target variables. It's a powerful technique that improves accuracy and reduces overfitting compared to single decision trees.

## How Random Forest Regression Works

1. **Bootstrap Aggregating (Bagging)**: Creates multiple subsets of the training data with replacement.
2. **Decision Tree Creation**: Builds a regression decision tree for each subset, using a random subset of features at each split.
3. **Averaging**: For regression, the final prediction is the average of predictions from all trees.

## Advantages

- Reduces overfitting compared to individual decision trees
- Handles high-dimensional data effectively
- Provides feature importance measures
- Robust to outliers
- Can handle non-linear relationships well

## Disadvantages

- Less interpretable than a single decision tree
- Computationally expensive for large datasets
- May overfit on noisy datasets
- Not as effective for extrapolation (predictions outside the range of training data)

## Implementation

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Generate a sample dataset
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Make predictions
y_pred = rf_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': range(X.shape[1]),
    'Importance': rf_regressor.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance.head(10))
```

## Hyperparameters

- `n_estimators`: Number of trees in the forest
- `max_depth`: Maximum depth of each tree
- `min_samples_split`: Minimum samples required to split a node
- `min_samples_leaf`: Minimum samples required in a leaf node
- `max_features`: Number of features to consider for the best split

## Applications

- Housing price prediction
- Stock price forecasting
- Sales forecasting
- Energy consumption prediction
- Weather forecasting