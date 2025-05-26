---
title: Support Vector Machines
sidebar_position: 6
description: Understanding Support Vector Machine algorithms for classification and regression
---

# Support Vector Machines (SVM)

Support Vector Machines (SVM) are powerful **supervised learning algorithms** used primarily for classification but also applicable to regression. They work by finding the optimal hyperplane that best separates data points of different classes with the maximum margin of separation.

## 1. Types of Support Vector Machines

### A. Support Vector Classification (SVC)
- Linear SVC: Uses a linear kernel for linearly separable data
- Non-linear SVC: Uses non-linear kernels (RBF, polynomial, sigmoid) for complex data

### B. Support Vector Regression (SVR)
- Predicts continuous values while maintaining the core principles of SVM
- Uses an epsilon-insensitive loss function that ignores errors within a certain distance

---

## 2. How Support Vector Machines Work

### Core Concepts:

1. **Hyperplane**: 
   - In 2D, it's a line; in 3D, a plane; and in higher dimensions, a hyperplane
   - Separates data points of different classes

2. **Margin**:
   - Distance between the hyperplane and the nearest data points (support vectors)
   - SVM aims to maximize this margin

3. **Support Vectors**:
   - Data points closest to the decision boundary
   - Critical for defining the hyperplane
   - Only these points affect the position of the hyperplane

4. **Kernel Trick**:
   - Transforms data into higher dimensions where it becomes linearly separable
   - Common kernels: Linear, Polynomial, Radial Basis Function (RBF), Sigmoid

### Mathematical Foundation:

SVM optimizes this objective function:
- Minimize: ½||w||² + C∑ξᵢ
- Subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ and ξᵢ ≥ 0 for all i

Where:
- w = weight vector
- b = bias
- C = regularization parameter
- ξᵢ = slack variables (allowing some misclassification)
- (xᵢ, yᵢ) = training examples and labels

---

## 3. Example Use Case: Face Recognition

### Scenario
A security system wants to automatically identify authorized personnel using facial features. It extracts various facial measurements and uses SVM to classify whether a person is authorized or unauthorized.

### Dataset Sample

| Eye Distance | Nose Length | Face Width | Jaw Angle | Authorized |
|-------------|------------|-----------|-----------|------------|
| 6.2 | 5.1 | 14.3 | 92° | Yes |
| 5.8 | 4.8 | 13.9 | 89° | Yes |
| 6.5 | 5.3 | 15.0 | 95° | No |
| 5.9 | 4.9 | 13.8 | 91° | Yes |
| 6.4 | 5.4 | 14.8 | 94° | No |

### SVM Approach

1. **Data Preprocessing**: Normalize facial measurements
2. **Kernel Selection**: Use RBF kernel for non-linear decision boundary
3. **Hyperparameter Tuning**: Optimize C (regularization) and gamma (kernel coefficient)
4. **Training**: SVM learns the optimal hyperplane
5. **Classification**: New faces are classified based on which side of the hyperplane they fall on

For a new face with measurements:
- Eye Distance: 6.1
- Nose Length: 5.0
- Face Width: 14.2
- Jaw Angle: 90°

The SVM calculates its position relative to the hyperplane and outputs a classification (Authorized/Unauthorized) and confidence score.

---

## 4. Advantages of SVM

✅ Effective in high-dimensional spaces, even when dimensions exceed samples

✅ Memory efficient, using only a subset of training points (support vectors)

✅ Versatile through different kernel functions for various decision boundaries

✅ Robust against overfitting, especially in high-dimensional spaces

✅ Finds the global minimum, avoiding local minima problems

---

## 5. Limitations

❌ Not well-suited for very large datasets due to quadratic training time complexity

❌ Performance degrades with noisy data and overlapping classes

❌ Requires careful tuning of hyperparameters (C, gamma, kernel)

❌ Doesn't directly provide probability estimates (requires additional calibration)

❌ Challenging to interpret compared to simpler models

---

## 6. Real-World Applications

👤 **Face Recognition** – Identifying individuals from facial features

📄 **Text Classification** – Categorizing documents by topic or sentiment

🧬 **Bioinformatics** – Protein classification and gene expression analysis

🖼️ **Image Classification** – Detecting objects or patterns in images

🩺 **Medical Diagnosis** – Classifying medical conditions based on test results

---

## 7. Implementation Example

```python
# Install Dependencies
# pip install numpy pandas scikit-learn matplotlib seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA  # For visualization

# Sample face recognition dataset
data = {
    "EyeDistance": [6.2, 5.8, 6.5, 5.9, 6.4, 6.1, 5.7, 6.3, 6.0, 6.6, 5.9, 6.2, 6.4, 5.8, 6.3, 6.0, 5.9, 6.5, 6.1, 6.2],
    "NoseLength": [5.1, 4.8, 5.3, 4.9, 5.4, 5.0, 4.7, 5.2, 4.9, 5.5, 4.8, 5.1, 5.3, 4.7, 5.2, 5.0, 4.9, 5.4, 5.1, 5.2],
    "FaceWidth": [14.3, 13.9, 15.0, 13.8, 14.8, 14.1, 13.7, 14.5, 14.0, 15.2, 13.9, 14.2, 14.7, 13.8, 14.6, 14.1, 13.9, 15.1, 14.2, 14.4],
    "JawAngle": [92, 89, 95, 91, 94, 90, 88, 93, 92, 96, 90, 91, 94, 89, 93, 91, 90, 95, 92, 93],
    "Authorized": ["Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "No", "Yes", "No", 
                   "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "No", "Yes", "No"]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode the target variable
df["Authorized_Encoded"] = df["Authorized"].map({"Yes": 1, "No": 0})

# Display the first few rows
print(df.head())

# Visualize data relationships
plt.figure(figsize=(12, 10))
sns.pairplot(df, hue="Authorized", vars=["EyeDistance", "NoseLength", "FaceWidth", "JawAngle"])
plt.suptitle("Facial Feature Relationships by Authorization Status", y=1.02)
plt.show()

# Prepare the data
X = df[["EyeDistance", "NoseLength", "FaceWidth", "JawAngle"]]
y = df["Authorized_Encoded"]

# Standardize features (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Create and train the SVM model
# Use Grid Search to find optimal hyperparameters
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]  # Probability of being authorized

# Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Unauthorized', 'Authorized'], yticklabels=['Unauthorized', 'Authorized'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Unauthorized', 'Authorized']))

# Visualize decision boundaries (using PCA for dimensionality reduction to 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Train SVM on the PCA-transformed data
svm_pca = SVC(kernel='rbf', C=best_model.C, gamma=best_model.gamma, probability=True)
svm_pca.fit(pca.transform(X_train), y_train)

# Create a mesh grid for plotting the decision boundary
h = 0.02  # step size in the mesh
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary and data points
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('SVM Decision Boundary (with PCA for visualization)')
plt.show()

# Make a prediction for a new face
new_face = np.array([[6.1, 5.0, 14.2, 90]])  # Eye Distance, Nose Length, Face Width, Jaw Angle
new_face_scaled = scaler.transform(new_face)
prediction = best_model.predict(new_face_scaled)
probability = best_model.predict_proba(new_face_scaled)

print(f"\nPrediction for new face:")
print(f"Decision: {'Authorized' if prediction[0] == 1 else 'Unauthorized'}")
print(f"Authorization Probability: {probability[0][1]:.4f}")
```

## Summary

✅ Built an SVM model for face recognition classification

✅ Learned how SVM finds the optimal hyperplane with maximum margin

✅ Used grid search to find the best hyperparameters

✅ Visualized the decision boundary in a reduced dimension space

✅ Evaluated model performance and made predictions for new faces