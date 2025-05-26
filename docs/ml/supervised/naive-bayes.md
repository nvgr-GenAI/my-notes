---
title: Naive Bayes
sidebar_position: 7
description: Understanding Naive Bayes algorithms for probabilistic classification
---

# Naive Bayes

Naive Bayes is a family of **probabilistic classifiers** based on applying Bayes' theorem with strong (naive) independence assumptions between the features. Despite its simplicity, Naive Bayes can be surprisingly effective, especially for text classification and when the independence assumption approximately holds.

## 1. Types of Naive Bayes

### A. Gaussian Naive Bayes
- Assumes features follow a normal distribution
- Used for continuous data
- Example: Medical diagnosis based on continuous measurements (blood pressure, temperature)

### B. Multinomial Naive Bayes
- Assumes features follow a multinomial distribution
- Commonly used for document classification
- Example: Email classification based on word frequencies

### C. Bernoulli Naive Bayes
- Features are binary (0/1)
- Used for text classification with binary word occurrence features
- Example: Spam detection where features are presence/absence of specific words

---

## 2. How Naive Bayes Works

Naive Bayes is based on **Bayes' theorem**:

P(y|X) = [P(X|y) × P(y)] / P(X)

Where:
- P(y|X) is the posterior probability of class y given predictor X
- P(X|y) is the likelihood of predictor X given class y
- P(y) is the prior probability of class y
- P(X) is the prior probability of predictor X

The "naive" assumption is that features are conditionally independent given the class:

P(X₁, X₂, ..., Xₙ|y) = P(X₁|y) × P(X₂|y) × ... × P(Xₙ|y)

This simplifies the calculation and makes the algorithm computationally efficient.

### Classification Rule:
Choose the class with the highest posterior probability:

y = argmax[y] P(y) × ∏ P(Xᵢ|y)

---

## 3. Example Use Case: Email Spam Detection

### Scenario
An email service wants to filter spam emails based on the presence of certain words.

### Dataset Sample

| Email | Contains "Money" | Contains "Offer" | Contains "Meeting" | Contains "Report" | Is Spam |
|-------|-----------------|-----------------|-------------------|------------------|---------|
| Email 1 | Yes | Yes | No | No | Yes |
| Email 2 | No | No | Yes | Yes | No |
| Email 3 | Yes | Yes | No | No | Yes |
| Email 4 | No | No | No | Yes | No |
| Email 5 | Yes | No | Yes | No | No |

### Naive Bayes Approach

1. **Calculate prior probabilities**: P(Spam) and P(Not Spam)
2. **Calculate likelihoods**: P(Word|Spam) and P(Word|Not Spam) for each word
3. **Apply Bayes' theorem** to get P(Spam|Words) and P(Not Spam|Words)
4. **Classify** based on which probability is higher

Let's calculate for a new email containing "Money" and "Offer":

From the training data:
- P(Spam) = 2/5 = 0.4
- P(Not Spam) = 3/5 = 0.6
- P("Money"|Spam) = 2/2 = 1.0
- P("Money"|Not Spam) = 1/3 = 0.33
- P("Offer"|Spam) = 2/2 = 1.0
- P("Offer"|Not Spam) = 0/3 = 0.0

Using Bayes' theorem (with Laplace smoothing to handle zero probabilities):

P(Spam|"Money", "Offer") ∝ P(Spam) × P("Money"|Spam) × P("Offer"|Spam) = 0.4 × 1.0 × 1.0 = 0.4

P(Not Spam|"Money", "Offer") ∝ P(Not Spam) × P("Money"|Not Spam) × P("Offer"|Not Spam) = 0.6 × 0.33 × 0.0 ≈ 0

Since P(Spam|"Money", "Offer") > P(Not Spam|"Money", "Offer"), the email is classified as spam.

---

## 4. Advantages of Naive Bayes

✅ Simple, fast, and efficient

✅ Works well with high-dimensional data (like text)

✅ Requires less training data than many algorithms

✅ Not sensitive to irrelevant features

✅ Handles missing values by ignoring them during calculation

---

## 5. Limitations

❌ Assumes strong independence between features (often not realistic)

❌ Can be outperformed by more sophisticated models

❌ Zero frequency problem (when a category has no occurrences of a feature)

❌ Sensitive to how input data is prepared

❌ Can't learn interactions between features

---

## 6. Real-World Applications

📧 **Email Spam Filtering** – Identifying junk/spam emails

📰 **News Categorization** – Classifying articles by topic

🎭 **Sentiment Analysis** – Determining positive/negative sentiment in text

📄 **Document Classification** – Organizing documents into categories

🩺 **Disease Diagnosis** – Classifying diseases based on symptoms

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
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

# Sample email dataset
emails = [
    {'text': 'Get money fast with this exclusive offer free cash', 'label': 'spam'},
    {'text': 'Meeting scheduled for tomorrow please bring your reports', 'label': 'not spam'},
    {'text': 'Earn money with this amazing offer limited time', 'label': 'spam'},
    {'text': 'Project report due next week team meeting tomorrow', 'label': 'not spam'},
    {'text': 'Free money guaranteed offer click now to claim', 'label': 'spam'},
    {'text': 'Reminder about performance review next week', 'label': 'not spam'},
    {'text': 'Win cash prizes in our exclusive offer', 'label': 'spam'},
    {'text': 'Notes from yesterday meeting check attachment', 'label': 'not spam'},
    {'text': 'Double your money guaranteed investment offer', 'label': 'spam'},
    {'text': 'Agenda for quarterly review meeting tomorrow', 'label': 'not spam'},
    {'text': 'Free cash offer limited time only', 'label': 'spam'},
    {'text': 'Please submit your expense reports by Friday', 'label': 'not spam'},
    {'text': 'Financial freedom with this exclusive money making offer', 'label': 'spam'},
    {'text': 'Team lunch scheduled for tomorrow noon', 'label': 'not spam'},
    {'text': 'Urgent offer cash prizes claim now money', 'label': 'spam'},
    {'text': 'New company policy document please review', 'label': 'not spam'},
    {'text': 'Discount offer free trial money back guarantee', 'label': 'spam'},
    {'text': 'Weekly meeting rescheduled to Thursday', 'label': 'not spam'},
    {'text': 'Make money fast home business offer', 'label': 'spam'},
    {'text': 'Quarterly report attached please review before meeting', 'label': 'not spam'}
]

# Convert to DataFrame
df = pd.DataFrame(emails)

# Display the first few rows
print(df.head())

# Create a binary label
df['is_spam'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)

# Count words that appear in spam vs. not spam emails
def count_words_in_class(df, word, class_value):
    mask = df['is_spam'] == class_value
    texts_in_class = df.loc[mask, 'text'].str.lower()
    count = sum(texts_in_class.str.contains(word.lower()))
    return count, len(texts_in_class)

# Calculate word frequencies for common spam/non-spam indicators
words = ['money', 'offer', 'meeting', 'report', 'free', 'cash', 'guarantee', 'review', 'team']
word_stats = []

for word in words:
    spam_count, total_spam = count_words_in_class(df, word, 1)
    not_spam_count, total_not_spam = count_words_in_class(df, word, 0)
    
    spam_freq = spam_count / total_spam
    not_spam_freq = not_spam_count / total_not_spam
    
    word_stats.append({
        'word': word,
        'spam_frequency': spam_freq,
        'not_spam_frequency': not_spam_freq,
        'frequency_ratio': spam_freq / not_spam_freq if not_spam_freq > 0 else float('inf')
    })

word_stats_df = pd.DataFrame(word_stats)
print("\nWord frequencies in spam vs. non-spam emails:")
print(word_stats_df.sort_values('frequency_ratio', ascending=False))

# Visualize word frequencies
plt.figure(figsize=(12, 6))
word_stats_df = word_stats_df.sort_values('word')
x = np.arange(len(word_stats_df))
width = 0.35

plt.bar(x - width/2, word_stats_df['spam_frequency'], width, label='Spam')
plt.bar(x + width/2, word_stats_df['not_spam_frequency'], width, label='Not Spam')

plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Word Frequencies in Spam vs. Non-Spam Emails')
plt.xticks(x, word_stats_df['word'], rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['is_spam'], test_size=0.3, random_state=42)

# Create a pipeline with CountVectorizer and Naive Bayes
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english')),  # Convert text to word count vectors
    ('classifier', MultinomialNB())  # Multinomial Naive Bayes classifier
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]  # Probability of being spam

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

# Extract feature names and their coefficients
feature_names = pipeline.named_steps['vectorizer'].get_feature_names_out()
coefficients = pipeline.named_steps['classifier'].feature_log_prob_

# Words that strongly indicate spam (highest log-probability difference)
log_prob_diffs = coefficients[1] - coefficients[0]  # Spam - Not Spam
strongest_spam_indices = log_prob_diffs.argsort()[-10:][::-1]  # Top 10 spam indicators
strongest_ham_indices = log_prob_diffs.argsort()[:10]  # Top 10 ham indicators

print("\nTop 10 words indicating spam:")
for idx in strongest_spam_indices:
    print(f"{feature_names[idx]}: {np.exp(coefficients[1][idx]):.4f} vs {np.exp(coefficients[0][idx]):.4f}")

print("\nTop 10 words indicating not spam:")
for idx in strongest_ham_indices:
    print(f"{feature_names[idx]}: {np.exp(coefficients[0][idx]):.4f} vs {np.exp(coefficients[1][idx]):.4f}")

# Test with new email examples
new_emails = [
    "Exclusive offer: Make money from home today!",
    "Team meeting tomorrow to discuss quarterly reports"
]

new_predictions = pipeline.predict(new_emails)
new_probabilities = pipeline.predict_proba(new_emails)[:, 1]

print("\nPredictions for new emails:")
for i, email in enumerate(new_emails):
    print(f"Email: {email}")
    print(f"Prediction: {'Spam' if new_predictions[i] == 1 else 'Not Spam'}")
    print(f"Spam Probability: {new_probabilities[i]:.4f}")
    print()
```

## Summary

✅ Built a Naive Bayes model for email spam detection

✅ Understood how Bayes' theorem is applied to classify text

✅ Visualized word frequency differences between spam and non-spam emails

✅ Used a classification pipeline with text vectorization and Multinomial Naive Bayes

✅ Identified key words that indicate spam or legitimate emails