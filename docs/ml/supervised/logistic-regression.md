---
title: Logistic Regression
sidebar_position: 3
description: Understanding Logistic Regression algorithm for binary and multi-class classification
---

# Logistic Regression

Logistic Regression is a **supervised learning algorithm** used for classification problems. Despite its name, it's used for classification, not regression. It predicts the probability of an observation belonging to a certain class.

## 1. Types of Logistic Regression

### A. Binary Logistic Regression
- Used when the dependent variable has only two possible outcomes (Yes/No, 0/1)
- Examples: Email spam detection (Spam/Not Spam), Disease diagnosis (Present/Absent)

### B. Multinomial Logistic Regression
- Used when the dependent variable has three or more unordered categories
- Example: Predicting the type of cuisine (Italian, Chinese, Mexican, etc.)

### C. Ordinal Logistic Regression
- Used when the dependent variable has three or more ordered categories
- Example: Movie ratings (1 star, 2 stars, 3 stars, etc.)

---

## 2. How Logistic Regression Works

Logistic Regression uses the **logistic (sigmoid) function** to transform a linear prediction into a probability between 0 and 1:

P(Y=1) = 1 / (1 + e^(-z))

Where:
- P(Y=1) is the probability that the observation belongs to class 1
- z is the linear combination of features: z = b₀ + b₁X₁ + b₂X₂ + ... + bₙXₙ
- e is the base of natural logarithm (approximately 2.71828)

The decision boundary is:
- If P(Y=1) ≥ 0.5, predict class 1
- If P(Y=1) < 0.5, predict class 0

The model is trained using **Maximum Likelihood Estimation**, which finds the parameters that maximize the likelihood of observing the given data.

---

## 3. Example Use Case: Email Spam Detection

### Scenario
An email service wants to classify incoming emails as spam or not spam based on features like:
- Number of uppercase words
- Number of suspicious phrases
- Presence of URL with different domain than sender
- Email length

### Dataset Sample

| Uppercase Words | Suspicious Phrases | Different Domain URL | Length (words) | Is Spam |
|----------------|-------------------|---------------------|---------------|---------|
| 10 | 5 | Yes | 150 | Yes |
| 2 | 0 | No | 200 | No |
| 15 | 8 | Yes | 50 | Yes |
| 3 | 1 | No | 180 | No |

### Applying Logistic Regression

The model learns the relationship:

P(Spam) = 1 / (1 + e^-(b₀ + b₁(Uppercase) + b₂(Suspicious) + b₃(URL) + b₄(Length)))

If the trained model gives coefficients:
- b₀ = -5.0 (intercept)
- b₁ = 0.4 (Uppercase Words)
- b₂ = 0.8 (Suspicious Phrases)
- b₃ = 2.5 (Different Domain URL, where 1=Yes, 0=No)
- b₄ = -0.01 (Length)

For a new email with:
- 12 uppercase words
- 6 suspicious phrases
- Contains a different domain URL (1)
- 100 words length

z = -5.0 + 0.4(12) + 0.8(6) + 2.5(1) + (-0.01)(100)
z = -5.0 + 4.8 + 4.8 + 2.5 - 1.0 = 6.1

P(Spam) = 1 / (1 + e^(-6.1)) ≈ 0.998

With a probability of 0.998, the email would be classified as spam.

---

## 4. Advantages of Logistic Regression

✅ Provides probabilities rather than just classifications

✅ Efficient to train and doesn't require high computational power

✅ Less prone to overfitting when regularization is used

✅ Highly interpretable - coefficients directly indicate feature importance and direction

---

## 5. Limitations

❌ Assumes linearity between independent variables and the log-odds of the outcome

❌ May underperform when there are complex relationships between variables

❌ Requires more data to achieve stability when there are many features

❌ Struggles with imbalanced datasets without proper adjustments

---

## 6. Real-World Applications

📧 **Email Spam Detection** – Classifying emails as spam or legitimate

🏥 **Disease Diagnosis** – Predicting the presence of a disease based on symptoms and test results

💳 **Credit Card Fraud Detection** – Identifying fraudulent transactions

📱 **Customer Churn Prediction** – Determining which customers are likely to leave a service

🗳️ **Voter Behavior Prediction** – Forecasting whether someone will vote for a particular candidate

---

## 7. Implementation Example

```python
# Install Dependencies
# pip install numpy pandas scikit-learn matplotlib seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

# Sample email spam dataset
data = {
    "Uppercase_Words": [10, 2, 15, 3, 20, 5, 12, 1, 8, 2, 18, 4, 7, 3, 9],
    "Suspicious_Phrases": [5, 0, 8, 1, 10, 2, 6, 0, 4, 1, 9, 0, 3, 1, 5],
    "Different_Domain_URL": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],  # 1=Yes, 0=No
    "Length": [150, 200, 50, 180, 100, 220, 90, 300, 120, 250, 80, 270, 180, 200, 150],
    "Is_Spam": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0]  # 1=Spam, 0=Not Spam
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the first few rows
print(df.head())

# Visualize correlations
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlations")
plt.show()

# Prepare the data
X = df[["Uppercase_Words", "Suspicious_Phrases", "Different_Domain_URL", "Length"]]
y = df["Is_Spam"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model with regularization
model = LogisticRegression(C=1.0, penalty='l2', solver='liblinear')
model.fit(X_train_scaled, y_train)

# Print model coefficients
print("Intercept:", model.intercept_[0])
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {coef}")

# Make predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probability of being spam

# Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Random prediction line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Make a prediction for a new email
new_email = np.array([[12, 6, 1, 100]])  # 12 uppercase words, 6 suspicious phrases, has different domain URL, 100 words long
new_email_scaled = scaler.transform(new_email)
spam_probability = model.predict_proba(new_email_scaled)[0, 1]

print(f"\nPrediction for new email:")
print(f"Spam probability: {spam_probability:.4f}")
print(f"Classification: {'Spam' if spam_probability >= 0.5 else 'Not Spam'}")
```

## Summary

✅ Built a Logistic Regression model for email spam detection

✅ Learned how the model uses the sigmoid function to predict probabilities

✅ Demonstrated coefficients interpretation for feature importance

✅ Evaluated the model using accuracy, confusion matrix, and ROC curve

✅ Applied the model to classify new emails as spam or not spam