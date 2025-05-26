---
title: K-Nearest Neighbors
sidebar_position: 8
description: Understanding the K-Nearest Neighbors (KNN) algorithm for classification and regression
---

# K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a simple, versatile, and intuitive **supervised learning algorithm** used for both classification and regression. It works on the principle that similar data points exist in close proximity to each other.

## 1. Types of K-Nearest Neighbors

### A. KNN for Classification
- Predicts the class of a data point based on the majority class among its K nearest neighbors
- Example: Classifying whether an email is spam based on similarities to known emails

### B. KNN for Regression
- Predicts a continuous value by averaging the values of its K nearest neighbors
- Example: Predicting house prices based on similar houses in the neighborhood

---

## 2. How KNN Works

KNN is a **lazy learning** algorithm, meaning it doesn't build a model during training but memorizes the training data. The algorithm follows these steps:

1. **Store** all training examples with their labels
2. **Calculate** the distance between a new example and all training examples
3. **Select** the K-nearest examples based on the calculated distances
4. For **classification**: Take a majority vote among the K neighbors
5. For **regression**: Calculate the average value among the K neighbors

### Distance Metrics:
Different distance measures can be used:

- **Euclidean Distance** (most common): √(Σ(xᵢ - yᵢ)²)
- **Manhattan Distance**: Σ|xᵢ - yᵢ|
- **Minkowski Distance**: (Σ|xᵢ - yᵢ|ᵖ)^(1/p)
- **Hamming Distance**: Count of positions where corresponding values differ (for categorical data)

### Choosing K:
- **Small K**: More sensitive to noise, more flexible decision boundary
- **Large K**: Smoother decision boundary, less prone to overfitting
- **Odd K**: Recommended for binary classification to avoid ties

---

## 3. Example Use Case: Iris Flower Classification

### Scenario
A botanist wants to classify iris flowers into species based on measurements of the sepal and petal dimensions.

### Dataset Sample

| Sepal Length (cm) | Sepal Width (cm) | Petal Length (cm) | Petal Width (cm) | Species |
|-------------------|------------------|-------------------|------------------|---------|
| 5.1 | 3.5 | 1.4 | 0.2 | Setosa |
| 7.0 | 3.2 | 4.7 | 1.4 | Versicolor |
| 6.3 | 3.3 | 6.0 | 2.5 | Virginica |
| 4.9 | 3.0 | 1.4 | 0.2 | Setosa |
| 6.4 | 3.2 | 4.5 | 1.5 | Versicolor |

### KNN Approach
Let's classify a new flower with measurements:
- Sepal Length: 5.8 cm
- Sepal Width: 2.7 cm
- Petal Length: 4.1 cm
- Petal Width: 1.0 cm

Steps:
1. Calculate distances from the new flower to all flowers in the dataset
2. With K=3, select the 3 closest flowers based on distance
3. Take a majority vote of the species from these 3 neighbors
4. Assign the majority species to the new flower

In this example, if 2 of the 3 nearest neighbors are Versicolor and 1 is Setosa, the new flower is classified as Versicolor.

---

## 4. Advantages of KNN

✅ Simple and intuitive algorithm

✅ No training phase (lazy learning)

✅ Works for both classification and regression

✅ No assumptions about data distribution (non-parametric)

✅ Naturally handles multi-class classification

---

## 5. Limitations

❌ Computationally expensive for large datasets (calculates distances to all points)

❌ Requires feature scaling for accurate results

❌ Struggles with high-dimensional data (curse of dimensionality)

❌ Sensitive to noisy data and outliers

❌ Imbalanced data can bias predictions toward the majority class

---

## 6. Real-World Applications

📱 **Recommendation Systems** – "Customers who bought this also bought..."

🎵 **Music Genre Classification** – Categorizing songs based on audio features

🏥 **Medical Diagnosis** – Classifying diseases based on symptom proximity to known cases

💳 **Credit Scoring** – Assessing creditworthiness based on similar applicants

🎮 **Computer Vision** – Image recognition and object detection

---

## 7. Implementation Example

```python
# Install Dependencies
# pip install numpy pandas scikit-learn matplotlib seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create a DataFrame for easier data manipulation and visualization
df = pd.DataFrame(X, columns=feature_names)
df['species'] = pd.Categorical.from_codes(y, target_names)

# Display the first few rows
print(df.head())

# Visualize the data
plt.figure(figsize=(15, 10))

# Pairplot to see relationships between features
sns.pairplot(df, hue='species', markers=["o", "s", "D"])
plt.suptitle("Iris Dataset - Feature Relationships by Species", y=1.02)
plt.show()

# Scatter plot of the most discriminative features
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=df['species'], style=df['species'], s=70)
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('Iris Species by Sepal Dimensions')

plt.subplot(1, 2, 2)
sns.scatterplot(x=df.iloc[:, 2], y=df.iloc[:, 3], hue=df['species'], style=df['species'], s=70)
plt.xlabel(feature_names[2])
plt.ylabel(feature_names[3])
plt.title('Iris Species by Petal Dimensions')

plt.tight_layout()
plt.show()

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Finding the optimal K value
k_values = list(range(1, 31))
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Plot the CV accuracy vs. K Value
plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, 'o-')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Finding Optimal K Value')
plt.grid(True)
plt.show()

# Get the optimal K value
optimal_k = k_values[cv_scores.index(max(cv_scores))]
print(f"The optimal number of neighbors is {optimal_k} with accuracy: {max(cv_scores):.4f}")

# Train the model with the optimal K
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Species')
plt.ylabel('Actual Species')
plt.show()

# Visualize the decision boundaries (using only 2 features for simplicity)
def plot_decision_boundaries(X, y, model, feature_idx=(2, 3)):
    # Extract the two features we want to visualize
    X_vis = X[:, feature_idx]
    
    # Create a mesh grid
    h = 0.02  # Step size
    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Create a test dataset containing only the two selected features
    X_test_mesh = np.c_[xx.ravel(), yy.ravel()]
    
    # We need to create a complete feature vector for prediction
    X_test_full = np.zeros((X_test_mesh.shape[0], X.shape[1]))
    X_test_full[:, feature_idx[0]] = X_test_mesh[:, 0]
    X_test_full[:, feature_idx[1]] = X_test_mesh[:, 1]
    
    # Predict the class for each point in the mesh
    Z = model.predict(X_test_full)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    
    # Plot the training points
    scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu)
    plt.xlabel(feature_names[feature_idx[0]])
    plt.ylabel(feature_names[feature_idx[1]])
    plt.title(f'KNN Decision Boundaries (K={optimal_k})')
    plt.legend(handles=scatter.legend_elements()[0], labels=target_names)
    plt.show()

# Train a KNN model on all data for visualization
knn_vis = KNeighborsClassifier(n_neighbors=optimal_k)
knn_vis.fit(X_scaled, y)

# Visualize the decision boundaries using petal features
plot_decision_boundaries(X_scaled, y, knn_vis, feature_idx=(2, 3))

# Classify a new iris flower
new_flower = np.array([[5.8, 2.7, 4.1, 1.0]])  # Values from our example
new_flower_scaled = scaler.transform(new_flower)
prediction = knn.predict(new_flower_scaled)
probabilities = knn.predict_proba(new_flower_scaled)

print(f"\nNew Flower Measurements:")
for i, name in enumerate(feature_names):
    print(f"{name}: {new_flower[0][i]} cm")

print(f"\nPredicted Species: {target_names[prediction[0]]}")
print(f"Prediction Probabilities:")
for i, species in enumerate(target_names):
    print(f"{species}: {probabilities[0][i]:.4f}")

# Find and display the K nearest neighbors to the new flower
distances, indices = knn.kneighbors(new_flower_scaled)

print(f"\n{optimal_k} Nearest Neighbors:")
for i in range(optimal_k):
    neighbor_idx = indices[0][i]
    distance = distances[0][i]
    neighbor = X[neighbor_idx]
    species = target_names[y[neighbor_idx]]
    print(f"Neighbor {i+1}: {species} - Distance: {distance:.4f}")
    for j, name in enumerate(feature_names):
        print(f"  {name}: {neighbor[j]} cm")
```

## Summary

✅ Built a K-Nearest Neighbors model for Iris flower classification

✅ Explored how to find the optimal K value using cross-validation

✅ Visualized decision boundaries to understand how KNN classifies data

✅ Evaluated model performance using accuracy and confusion matrix

✅ Demonstrated how to identify the nearest neighbors for a new data point