# SVM Regression

Support Vector Machine (SVM) Regression is a powerful **supervised learning algorithm** that performs regression by finding the optimal hyperplane that maximizes the margin while keeping the error within a specified threshold. Also known as SVR (Support Vector Regression), it's particularly effective for non-linear regression tasks.

## How SVM Regression Works

Unlike SVM Classification, which tries to find a hyperplane that maximizes the margin between classes, SVM Regression tries to fit a hyperplane to the data that includes as many points as possible within a certain margin (epsilon ε). The key concepts include:

1. **Epsilon-Tube**: A tube of width ε around the hyperplane where errors are not penalized
2. **Support Vectors**: The data points that lie outside the ε-tube
3. **Kernel Trick**: A method to handle non-linear relationships by mapping the data to a higher-dimensional space

## Types of SVM Regression

1. **Epsilon-SVR**: Uses an ε-insensitive loss function which ignores errors within ε distance of the true value
2. **Nu-SVR**: Uses a parameter ν to control the number of support vectors and training errors

## Advantages

- Robust to outliers due to the ε-insensitive loss function
- Effective in high-dimensional spaces
- Versatile due to different kernel functions
- Memory efficient (uses only support vectors)
- Works well when the number of features is greater than the number of samples

## Disadvantages

- Computationally intensive for large datasets
- Requires careful tuning of hyperparameters
- Less interpretable than simpler models
- Finding the optimal values for regularization and ε can be challenging
- Not directly suitable for online learning

## Implementation

```python
from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Generate a sample regression dataset
X, y = make_regression(n_samples=200, n_features=1, noise=0.5, random_state=42)

# Add non-linearity to the data
y = y + np.sin(X[:, 0] * 2) * 5

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scale the target variable
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# Create and train SVM models with different kernels
kernels = ['linear', 'poly', 'rbf']
plt.figure(figsize=(18, 6))

for i, kernel in enumerate(kernels):
    # Create and train the model
    svr_model = SVR(kernel=kernel, gamma='auto', C=10, epsilon=0.1)
    svr_model.fit(X_train_scaled, y_train_scaled)
    
    # Make predictions
    y_pred_scaled = svr_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Plot results
    plt.subplot(1, 3, i+1)
    
    # Sort points for plot
    sort_idx = np.argsort(X.ravel())
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    # Create a mesh grid for predictions across the range
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_plot_scaled = scaler_X.transform(X_plot)
    y_plot_scaled = svr_model.predict(X_plot_scaled)
    y_plot = scaler_y.inverse_transform(y_plot_scaled.reshape(-1, 1)).ravel()
    
    # Plot the original data points
    plt.scatter(X, y, color='blue', s=30, alpha=0.5, label='Data points')
    plt.plot(X_plot, y_plot, color='red', linewidth=2, label=f'SVR with {kernel} kernel')
    
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title(f'SVR with {kernel} kernel\nMSE: {mse:.2f}, R²: {r2:.2f}')
    plt.legend()

plt.tight_layout()
plt.show()

# Fine-tuning SVR with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'epsilon': [0.01, 0.1, 0.5, 1.0],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train_scaled, y_train_scaled)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score (negative MSE): {grid_search.best_score_:.4f}")

# Train final model with best parameters
best_svr = grid_search.best_estimator_
y_pred_scaled = best_svr.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# Evaluate final model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Final model - MSE: {mse:.4f}, R²: {r2:.4f}")
```

## Kernel Functions

1. **Linear**: $K(x_i, x_j) = x_i^T x_j$
   - Suitable for linearly separable data
   
2. **Polynomial**: $K(x_i, x_j) = (γx_i^T x_j + r)^d$
   - Good for non-linear relationships
   - Parameters: degree $d$, gamma $γ$, and coefficient $r$
   
3. **RBF (Radial Basis Function)**: $K(x_i, x_j) = exp(-γ||x_i - x_j||^2)$
   - Most commonly used kernel for non-linear relationships
   - Works well for most datasets
   - Parameter: gamma $γ$ controls the influence of training examples
   
4. **Sigmoid**: $K(x_i, x_j) = tanh(γx_i^T x_j + r)$
   - Related to neural networks
   - Parameters: gamma $γ$ and coefficient $r$

## Hyperparameters

- **C**: Regularization parameter. Controls the trade-off between model complexity and the degree to which deviations larger than ε are tolerated.
- **epsilon (ε)**: Specifies the width of the ε-tube. Points inside the tube do not contribute to the regression fit.
- **gamma**: Kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels. Defines how far the influence of a single training example reaches.
- **kernel**: Type of kernel function to use.
- **degree**: Degree of the polynomial kernel function.

## Applications

- Financial market prediction
- Time series forecasting
- Environmental data analysis
- Energy consumption prediction
- Chemical process modeling