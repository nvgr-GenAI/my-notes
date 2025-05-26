---
title: Decision Tree Classification
sidebar_position: 3
description: Understanding Decision Tree algorithms for classification tasks
---

# Decision Tree Classification

Decision Trees for classification are **supervised learning algorithms** used to predict categorical target variables. They work by creating a model that predicts the class of a target variable by learning simple decision rules inferred from the data features.

## 1. How Classification Trees Work

Decision trees use a **hierarchical, tree-like structure** where:
- **Root Node**: Starting point, represents entire dataset
- **Decision Nodes**: Points where data is split based on feature values
- **Leaf Nodes**: Terminal nodes that provide the classification output

The algorithm works by:
1. **Selecting the best feature** to split the data at each node
2. **Creating child nodes** based on the split
3. **Repeating recursively** until stopping criteria are met

### Splitting Criteria:
- **Gini Impurity**: Measures the probability of incorrect classification
- **Entropy**: Measures the level of disorder or uncertainty
- **Information Gain**: The decrease in entropy after a split

### Stopping Criteria:
- Maximum depth reached
- Minimum samples required for a split
- Minimum samples required at a leaf node
- Maximum leaf nodes reached

---

## 2. Example Use Case: Loan Approval Prediction

### Scenario
A bank wants to predict whether to approve loan applications based on applicant characteristics.

### Dataset Sample

| Age | Income | Owns House | Credit Score | Loan Approved |
|-----|--------|------------|--------------|---------------|
| 25  | $45K   | No         | Good         | No            |
| 42  | $120K  | Yes        | Excellent    | Yes           |
| 30  | $70K   | No         | Good         | Yes           |
| 35  | $50K   | No         | Poor         | No            |
| 60  | $80K   | Yes        | Fair         | No            |

### Decision Tree Approach

The algorithm will:
1. Select the most informative feature for the first split (e.g., Credit Score)
2. Split the data based on this feature
3. Continue splitting recursively on each subset
4. Form a tree structure that can predict loan approval for new applicants

For example, the resulting tree might look like:
```
Credit Score?
├── Poor → Deny Loan
├── Fair 
│   ├── Income > $100K? → Approve Loan
│   └── Income ≤ $100K? → Deny Loan
├── Good
│   ├── Owns House? → Approve Loan
│   └── Renting
│       ├── Income > $60K? → Approve Loan
│       └── Income ≤ $60K? → Deny Loan
└── Excellent → Approve Loan
```

For a new applicant with:
- Credit Score: Good
- Owns House: No
- Income: $75K

We would trace the path: Good → Renting → Income > $60K → Approve Loan

---

## 3. Advantages of Decision Trees for Classification

✅ Simple to understand and visualize class boundaries

✅ Requires little data preprocessing (no normalization needed)

✅ Can handle both numerical and categorical data

✅ Implicitly performs feature selection

✅ Non-parametric (no assumptions about data distribution)

✅ Can handle multi-class problems naturally

---

## 4. Limitations

❌ Prone to overfitting, especially with deep trees

❌ Can be unstable (small variations in data can produce very different trees)

❌ May create biased trees if classes are imbalanced

❌ Greedy algorithms may not find the globally optimal tree

Techniques like **pruning**, **setting maximum depth**, and **ensemble methods** (Random Forests) can address some of these limitations.

---

## 5. Real-World Applications

🏥 **Medical Diagnosis** – Predicting diseases based on symptoms

📊 **Customer Segmentation** – Grouping customers based on behavior

💳 **Credit Risk Assessment** – Determining credit approval

🤖 **Sentiment Analysis** – Classifying text sentiment

🎯 **Target Marketing** – Identifying likely customers for campaigns

---

## 6. Implementation Example

```python
# Install Dependencies
# pip install numpy pandas scikit-learn matplotlib seaborn graphviz pydotplus

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import graphviz

# Sample loan approval dataset
data = {
    'Age': [25, 42, 30, 35, 60, 48, 33, 27, 45, 52, 38, 29, 36, 55, 22, 43, 50, 31, 28, 40],
    'Income': [45000, 120000, 70000, 50000, 80000, 90000, 65000, 35000, 110000, 75000, 
               60000, 40000, 85000, 95000, 30000, 100000, 130000, 55000, 47000, 72000],
    'OwnsHouse': ['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 
                  'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes'],
    'CreditScore': ['Good', 'Excellent', 'Good', 'Poor', 'Fair', 'Good', 'Excellent', 'Fair', 'Good', 'Excellent',
                    'Poor', 'Fair', 'Good', 'Fair', 'Poor', 'Excellent', 'Good', 'Fair', 'Good', 'Good'],
    'LoanApproved': ['No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes',
                     'No', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the first few rows
print(df.head())

# Convert categorical to numerical values
le_owns_house = LabelEncoder()
le_credit_score = LabelEncoder()
le_loan_approved = LabelEncoder()

df['OwnsHouse_Encoded'] = le_owns_house.fit_transform(df['OwnsHouse'])
df['CreditScore_Encoded'] = le_credit_score.fit_transform(df['CreditScore'])
df['LoanApproved_Encoded'] = le_loan_approved.fit_transform(df['LoanApproved'])

# Map the encoded values back to their original categories for better interpretation
credit_mapping = dict(zip(le_credit_score.transform(le_credit_score.classes_), le_credit_score.classes_))
print("\nCredit Score Mapping:", credit_mapping)

# Basic data exploration
print("\nData Summary:")
print(df.describe())

print("\nLoan Approval Distribution:")
print(df['LoanApproved'].value_counts())

# Visualize correlations
plt.figure(figsize=(10, 8))
# Select only numerical columns
numeric_df = df[['Age', 'Income', 'OwnsHouse_Encoded', 'CreditScore_Encoded', 'LoanApproved_Encoded']]
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Visualize the relationship between features and loan approval
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
sns.boxplot(x='LoanApproved', y='Age', data=df)
plt.title('Age vs. Loan Approval')

plt.subplot(2, 2, 2)
sns.boxplot(x='LoanApproved', y='Income', data=df)
plt.title('Income vs. Loan Approval')

plt.subplot(2, 2, 3)
sns.countplot(x='OwnsHouse', hue='LoanApproved', data=df)
plt.title('Home Ownership vs. Loan Approval')

plt.subplot(2, 2, 4)
sns.countplot(x='CreditScore', hue='LoanApproved', data=df)
plt.title('Credit Score vs. Loan Approval')

plt.tight_layout()
plt.show()

# Prepare data for the model
X = df[['Age', 'Income', 'OwnsHouse_Encoded', 'CreditScore_Encoded']]
y = df['LoanApproved_Encoded']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a Decision Tree model
dt_classifier = DecisionTreeClassifier(random_state=42)

# Cross-validation to find optimal max_depth
depth_range = range(1, 10)
accuracy_scores = []

for depth in depth_range:
    dt_classifier = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(dt_classifier, X_train, y_train, cv=5, scoring='accuracy')
    accuracy_scores.append(scores.mean())

# Plot cross-validation results
plt.figure(figsize=(10, 6))
plt.plot(depth_range, accuracy_scores, marker='o')
plt.xlabel('Maximum Depth')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Finding Optimal Tree Depth')
plt.grid(True)
plt.show()

# Get the best depth
best_depth = depth_range[np.argmax(accuracy_scores)]
print(f"\nOptimal tree depth: {best_depth}")

# Train the final model with the best depth
dt_classifier = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred = dt_classifier.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
target_names = ['Denied', 'Approved']
print(classification_report(y_test, y_pred, target_names=target_names))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Visualize the decision tree
plt.figure(figsize=(20, 10))
feature_names = ['Age', 'Income', 'Owns House', 'Credit Score']
class_names = ['Denied', 'Approved']
plot_tree(dt_classifier, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
plt.title(f"Decision Tree (Max Depth = {best_depth})")
plt.show()

# For a more detailed visualization using graphviz
dot_data = export_graphviz(dt_classifier, out_file=None, 
                           feature_names=feature_names,
                           class_names=class_names,
                           filled=True, rounded=True,  
                           special_characters=True)
graph = graphviz.Source(dot_data)
# graph.render("loan_decision_tree") # Uncomment to save the visualization

# Feature importance
importances = dt_classifier.feature_importances_
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

# Predict on new data
new_applicant = np.array([[30, 75000, 0, le_credit_score.transform(['Good'])[0]]])
prediction = dt_classifier.predict(new_applicant)
probability = dt_classifier.predict_proba(new_applicant)

print("\nNew Applicant:")
print(f"Age: 30")
print(f"Income: $75,000")
print(f"Owns House: No")
print(f"Credit Score: Good")
print(f"\nLoan Decision: {'Approved' if prediction[0] == 1 else 'Denied'}")
print(f"Probability: Denied: {probability[0][0]:.4f}, Approved: {probability[0][1]:.4f}")

# Find the path through the decision tree for the new applicant
def get_decision_path(tree, X, feature_names):
    node_indicator = tree.decision_path(X)
    leaf_id = tree.apply(X)
    
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    
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
                
            # Format the rule for better readability
            if feature[node_id] == 2:  # OwnsHouse
                value = "Yes" if threshold[node_id] < 0.5 else "No"
                rules.append(f"{feature_names[feature[node_id]]} = {value}")
            elif feature[node_id] == 3:  # CreditScore
                value = list(credit_mapping.values())[int(threshold[node_id])]
                rules.append(f"{feature_names[feature[node_id]]} = {value}")
            else:
                rules.append(f"{feature_names[feature[node_id]]} {threshold_sign} {threshold[node_id]:.2f}")
    
    return rules

decision_path = get_decision_path(dt_classifier, new_applicant, feature_names)
print("\nDecision Path:")
for i, rule in enumerate(decision_path):
    print(f"{i+1}. {rule}")
```

## Summary

✅ Built a Decision Tree classification model for loan approval prediction

✅ Used cross-validation to find the optimal tree depth

✅ Visualized the decision tree for interpretability

✅ Analyzed feature importance to understand key factors

✅ Demonstrated making and explaining predictions for new applicants with classification probabilities