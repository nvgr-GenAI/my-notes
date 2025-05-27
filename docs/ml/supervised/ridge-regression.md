# Ridge Regression

Ridge Regression is a regularized version of **Linear Regression** that adds an L2 penalty term to the cost function. This technique helps prevent overfitting and handles multicollinearity by shrinking the coefficient estimates toward zero.

## How Ridge Regression Works

Ridge Regression minimizes the following objective function:

<div class="math">
min<sub>β</sub> Σ<sub>i=1</sub><sup>n</sup>(y<sub>i</sub> - β<sub>0</sub> - Σ<sub>j=1</sub><sup>p</sup>β<sub>j</sub>x<sub>ij</sub>)<sup>2</sup> + λΣ<sub>j=1</sub><sup>p</sup>β<sub>j</sub><sup>2</sup>
</div>

Where:
- y<sub>i</sub> is the target variable
- x<sub>ij</sub> are the features
- β<sub>0</sub>, β<sub>j</sub> are the coefficients
- λ is the regularization parameter
- The term λΣ<sub>j=1</sub><sup>p</sup>β<sub>j</sub><sup>2</sup> is the L2 regularization penalty

## Advantages

- Reduces model complexity and prevents overfitting
- Effective for handling multicollinearity
- Can improve model generalization
- Coefficients shrink toward zero but rarely become exactly zero
- Stable when features are highly correlated

## Disadvantages

- Not suitable for feature selection (all features remain in the model)
- Less interpretable than simple linear regression
- Requires tuning of the regularization parameter
- May underfit with very high regularization
- Performance depends heavily on feature scaling

## Implementation

```python
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_regression(n_samples=200, n_features=50, noise=0.5, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Ridge model
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
plt.figure(figsize=(12, 8))

for i, alpha in enumerate(alphas):
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = ridge_model.predict(X_test_scaled)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Alpha: {alpha}, MSE: {mse:.4f}, R²: {r2:.4f}")
    
    # Plot coefficients
    plt.subplot(2, 3, i+1)
    plt.stem(range(X_train.shape[1]), ridge_model.coef_)
    plt.xlabel('Feature')
    plt.ylabel('Coefficient Value')
    plt.title(f'Alpha = {alpha}\nMSE = {mse:.4f}, R² = {r2:.4f}')

plt.tight_layout()
plt.show()

# Cross-validation to find optimal alpha
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

# Define range of alphas and cross-validation strategy
alphas = np.logspace(-3, 3, 50)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Create and fit RidgeCV model
ridge_cv = RidgeCV(alphas=alphas, cv=cv, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train_scaled, y_train)

# Print optimal alpha
print(f"Optimal Alpha: {ridge_cv.alpha_}")

# Train final model with optimal alpha
optimal_ridge = Ridge(alpha=ridge_cv.alpha_)
optimal_ridge.fit(X_train_scaled, y_train)
y_pred = optimal_ridge.predict(X_test_scaled)

# Evaluate final model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Final Model - MSE: {mse:.4f}, R²: {r2:.4f}")
```

## Selecting the Regularization Parameter (α)

The regularization parameter α controls the strength of the regularization:
- **Small α**: Close to ordinary least squares, might not solve multicollinearity issues
- **Large α**: More regularization, coefficients approach zero, might underfit

Common methods to select optimal α include:
- Cross-validation
- Information criteria (AIC, BIC)
- Domain knowledge

## Applications

- Financial market prediction
- Housing price prediction
- Genomic data analysis
- Research with many correlated predictors
- Any regression context where features exhibit multicollinearity