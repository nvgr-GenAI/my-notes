# Polynomial Regression

Polynomial Regression is an extension of **Linear Regression** that allows modeling of non-linear relationships between dependent and independent variables. It fits a polynomial equation to the data by transforming the original features into polynomial features.

## How Polynomial Regression Works

Polynomial Regression transforms the original features into polynomial features and then applies standard linear regression. For example, with a single input variable x, a polynomial regression of degree 2 would create the model:

y = β₀ + β₁x + β₂x² + ε

Where:
- β₀, β₁, β₂ are the coefficients
- x is the independent variable
- ε is the error term

## Advantages

- Can capture non-linear relationships
- Easy to understand and implement
- More flexible than simple linear regression
- Based on the well-established linear regression framework
- Useful for modeling curved relationships

## Disadvantages

- Prone to overfitting, especially with high-degree polynomials
- Sensitive to outliers
- Less interpretable with higher degrees
- May perform poorly on extrapolation
- Computationally expensive with many features due to combinatorial explosion

## Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Create polynomial regression models with different degrees
degrees = [1, 2, 3, 5]
plt.figure(figsize=(14, 8))

for i, degree in enumerate(degrees):
    # Create polynomial regression pipeline
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    # Fit the model
    model.fit(X, y)
    
    # Make predictions on a fine-grained scale for plotting
    X_test = np.linspace(0, 5, 100)[:, np.newaxis]
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    y_train_pred = model.predict(X)
    mse = mean_squared_error(y, y_train_pred)
    r2 = r2_score(y, y_train_pred)
    
    # Plot the results
    plt.subplot(2, 2, i+1)
    plt.scatter(X, y, color='blue', s=30, alpha=0.5, label='Data points')
    plt.plot(X_test, y_pred, color='red', label=f'Degree {degree}')
    plt.title(f'Polynomial Degree {degree}\nMSE = {mse:.4f}, R² = {r2:.4f}')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('y')

plt.tight_layout()
plt.show()
```

## Choosing the Right Polynomial Degree

The degree of the polynomial is a crucial hyperparameter:
- **Low degree (1-2)**: May underfit if the relationship is complex
- **Medium degree (3-5)**: Often provides a good balance for many real-world applications
- **High degree (>5)**: Likely to overfit the training data

Use techniques like cross-validation or information criteria (AIC, BIC) to select the optimal degree.

## Applications

- Economic trend analysis
- Growth curve modeling
- Physical phenomena with known polynomial behavior
- Signal processing
- Machine learning feature engineering