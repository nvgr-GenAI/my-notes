# SVM Classification

Support Vector Machine (SVM) Classification is a powerful **supervised learning algorithm** that finds the optimal hyperplane to separate different classes of data points. It's particularly effective for complex classification tasks in high-dimensional spaces.

## How SVM Classification Works

SVM classification works by finding the hyperplane that maximizes the margin between classes. The key concepts include:

1. **Optimal Hyperplane**: The decision boundary that maximizes the margin between classes
2. **Support Vectors**: The data points closest to the hyperplane that influence its position and orientation
3. **Margin**: The distance between the hyperplane and the closest data points from each class
4. **Kernel Trick**: A method to handle non-linearly separable data by mapping it to a higher-dimensional space

## Types of SVM Classification

1. **Linear SVM**: Uses a linear decision boundary
2. **Non-linear SVM**: Uses kernel functions (polynomial, RBF, sigmoid) for complex decision boundaries
3. **Multi-class SVM**: Extends the binary classification to multiple classes using either one-vs-rest or one-vs-one approaches

## Advantages

- Effective in high-dimensional spaces
- Memory efficient (uses only support vectors for the decision function)
- Versatile due to different kernel functions
- Works well with clear margin of separation
- Robust against overfitting, especially in high-dimensional spaces

## Disadvantages

- Not suitable for large datasets (computationally intensive)
- Performs poorly with overlapping classes
- Sensitive to noise
- Not directly probabilistic (requires adjustments for probability estimates)
- Requires careful tuning of hyperparameters

## Implementation

```python
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Generate a simple dataset for visualization
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, 
                           n_clusters_per_class=1, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train SVM models with different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
plt.figure(figsize=(15, 10))

for i, kernel in enumerate(kernels):
    # Create and train the model
    svm_model = svm.SVC(kernel=kernel, gamma='auto')
    svm_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = svm_model.predict(X_test_scaled)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    
    # Plot decision boundary
    plt.subplot(2, 2, i+1)
    
    # Create a mesh grid
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Scale the mesh grid
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_scaled = scaler.transform(mesh_points)
    
    # Plot the decision boundary
    Z = svm_model.predict(mesh_points_scaled)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    
    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'SVM with {kernel} kernel\nAccuracy: {accuracy:.4f}')

plt.tight_layout()
plt.show()

# Fine-tuning SVM with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(svm.SVC(), param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Train final model with best parameters
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_scaled)

# Evaluate final model
accuracy = accuracy_score(y_test, y_pred)
print(f"Final model accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

## Kernel Functions

1. **Linear**: $K(x_i, x_j) = x_i^T x_j$
   - Suitable for linearly separable data
   
2. **Polynomial**: $K(x_i, x_j) = (γx_i^T x_j + r)^d$
   - Good for non-linear boundaries
   - Parameters: degree $d$, gamma $γ$, and coefficient $r$
   
3. **RBF (Radial Basis Function)**: $K(x_i, x_j) = exp(-γ||x_i - x_j||^2)$
   - Most commonly used kernel
   - Works well for most datasets
   - Parameter: gamma $γ$ controls the influence of training examples
   
4. **Sigmoid**: $K(x_i, x_j) = tanh(γx_i^T x_j + r)$
   - Related to neural networks
   - Parameters: gamma $γ$ and coefficient $r$

## Hyperparameters

- **C**: Regularization parameter. Lower values = more regularization
- **gamma**: Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
- **kernel**: Type of kernel function
- **degree**: Degree of polynomial kernel function
- **coef0**: Independent term in kernel function

## Applications

- Image classification
- Text categorization
- Bioinformatics (protein classification, cancer classification)
- Face detection
- Handwriting recognition