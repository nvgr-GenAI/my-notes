# Lasso Regression

Lasso (Least Absolute Shrinkage and Selection Operator) Regression is a regularized version of **Linear Regression** that adds an L1 penalty term to the cost function. This technique performs both variable selection and regularization, effectively shrinking some coefficients exactly to zero.

## How Lasso Regression Works

Lasso Regression minimizes the following objective function:

$$ \min_{\beta} \sum_{i=1}^{n}(y_i - \beta_0 - \sum_{j=1}^{p}\beta_j x_{ij})^2 + \lambda \sum_{j=1}^{p}|\beta_j| $$

Where:
- $y_i$ is the target variable
- $x_{ij}$ are the features
- $\beta_0, \beta_j$ are the coefficients
- $\lambda$ is the regularization parameter
- The term $\lambda \sum_{j=1}^{p}|\beta_j|$ is the L1 regularization penalty

## Advantages

- Performs feature selection by shrinking coefficients exactly to zero
- Reduces model complexity and prevents overfitting
- Creates sparse models with few active coefficients
- Improves model interpretability through feature reduction
- Effective for high-dimensional datasets

## Disadvantages

- May remove relevant features if regularization is too strong
- Not stable with highly correlated features (tends to pick one)
- Requires careful tuning of the regularization parameter
- May underfit with very high regularization
- Performance depends heavily on feature scaling

## Implementation

```python
from sklearn.linear_model import Lasso
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

# Create and train the Lasso model with different alpha values
alphas = [0.001, 0.01, 0.1, 0.5, 1, 10]
plt.figure(figsize=(12, 8))

for i, alpha in enumerate(alphas):
    lasso_model = Lasso(alpha=alpha, max_iter=10000)
    lasso_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = lasso_model.predict(X_test_scaled)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Count non-zero coefficients
    n_nonzero = np.sum(lasso_model.coef_ != 0)
    
    print(f"Alpha: {alpha}, Non-zero coefficients: {n_nonzero}, MSE: {mse:.4f}, R²: {r2:.4f}")
    
    # Plot coefficients
    plt.subplot(2, 3, i+1)
    plt.stem(range(X_train.shape[1]), lasso_model.coef_)
    plt.xlabel('Feature')
    plt.ylabel('Coefficient Value')
    plt.title(f'Alpha = {alpha}\nNon-zero coef: {n_nonzero}/{X_train.shape[1]}')

plt.tight_layout()
plt.show()

# Cross-validation to find optimal alpha
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

# Define range of alphas and cross-validation strategy
alphas = np.logspace(-4, 1, 50)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Create and fit LassoCV model
lasso_cv = LassoCV(alphas=alphas, cv=cv, random_state=42, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

# Print optimal alpha
print(f"Optimal Alpha: {lasso_cv.alpha_}")

# Train final model with optimal alpha
optimal_lasso = Lasso(alpha=lasso_cv.alpha_, max_iter=10000)
optimal_lasso.fit(X_train_scaled, y_train)
y_pred = optimal_lasso.predict(X_test_scaled)

# Evaluate final model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
n_nonzero = np.sum(optimal_lasso.coef_ != 0)

print(f"Final Model - Non-zero coefficients: {n_nonzero}/{X_train.shape[1]}")
print(f"Final Model - MSE: {mse:.4f}, R²: {r2:.4f}")
```

## Selecting the Regularization Parameter (α)

The regularization parameter α controls the strength of the regularization:
- **Small α**: Close to ordinary least squares, most features retained
- **Large α**: More regularization, more coefficients shrink to zero

Common methods to select optimal α include:
- Cross-validation
- Information criteria (AIC, BIC)
- Domain knowledge

## Applications

- Feature selection in high-dimensional datasets
- Genomics and proteomics analysis
- Medical data analysis with many potential predictors
- Financial modeling with many potential factors
- Signal processing with sparse representation