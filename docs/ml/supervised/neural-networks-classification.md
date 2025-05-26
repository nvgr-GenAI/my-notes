# Neural Networks for Classification

Neural Networks for Classification are powerful supervised learning models that can learn complex patterns in data to categorize inputs into discrete classes. These networks excel at classification tasks ranging from simple binary decisions to complex multi-class problems.

## How Neural Networks Perform Classification

Neural networks for classification typically:

1. **Process input features** through multiple hidden layers
2. **Transform data** using weights, biases, and activation functions
3. **Output class probabilities** using a specialized output layer:
   - **Binary classification**: Single output neuron with sigmoid activation (0-1 range)
   - **Multi-class classification**: Multiple output neurons with softmax activation (probabilities sum to 1)

## Neural Network Architecture for Classification

![Neural Network Classification Architecture](https://i.imgur.com/JMfFnQE.png)

- **Input Layer**: One neuron per feature in your dataset
- **Hidden Layers**: Multiple layers with varying numbers of neurons to learn complex patterns
- **Output Layer**: 
  - For binary classification: 1 neuron with sigmoid activation
  - For multi-class classification: N neurons (one per class) with softmax activation

## Key Components for Classification

1. **Loss Functions**:
   - Binary classification: Binary cross-entropy
   - Multi-class classification: Categorical cross-entropy

2. **Evaluation Metrics**:
   - Accuracy
   - Precision and Recall
   - F1-Score
   - ROC AUC
   - Confusion Matrix

3. **Output Layer Design**:
   - Binary: `Dense(1, activation='sigmoid')`
   - Multi-class: `Dense(num_classes, activation='softmax')`

## Implementation: Binary Classification Example

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Generate a sample binary classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                           n_redundant=5, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a neural network for binary classification
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Display model architecture
model.summary()

# Train the model
history = model.fit(X_train_scaled, y_train, 
                    epochs=50, 
                    batch_size=32, 
                    validation_split=0.2, 
                    verbose=1)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluate the model
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot ROC curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
```

## Implementation: Multi-class Classification Example

```python
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical

# Load Iris dataset for multi-class classification
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert labels to one-hot encoding
y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)

# Create a neural network for multi-class classification
model_multiclass = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')  # Multi-class classification output (3 classes)
])

# Compile the model
model_multiclass.compile(optimizer=Adam(learning_rate=0.001), 
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])

# Display model architecture
model_multiclass.summary()

# Train the model
history = model_multiclass.fit(X_train_scaled, y_train_onehot, 
                               epochs=50, 
                               batch_size=16, 
                               validation_split=0.2, 
                               verbose=1)

# Evaluate the model
y_pred_proba = model_multiclass.predict(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

## Handling Imbalanced Datasets

Neural networks can struggle with imbalanced datasets. Common strategies include:

1. **Class weights**: Assign higher weights to minority classes
   ```python
   # Calculate class weights
   from sklearn.utils.class_weight import compute_class_weight
   class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
   class_weight_dict = dict(enumerate(class_weights))
   
   # Use in model training
   model.fit(X_train, y_train, class_weight=class_weight_dict, ...)
   ```

2. **Resampling**: Either undersample the majority class or oversample the minority class
   ```python
   from imblearn.over_sampling import SMOTE
   smote = SMOTE(random_state=42)
   X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
   ```

3. **Different metrics**: Focus on precision, recall, F1-score, or AUC instead of accuracy

## Hyperparameter Tuning for Classification

Key hyperparameters to tune:

- **Network architecture**: Number of layers and neurons
- **Learning rate**: Typically between 0.1 and 0.0001
- **Batch size**: Common values: 16, 32, 64, 128
- **Activation functions**: ReLU, LeakyReLU, ELU
- **Regularization**: L1, L2, or both (ElasticNet)
- **Dropout rate**: Typically between 0.2 and 0.5

```python
# Example of hyperparameter grid search
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def create_model(learning_rate=0.001, neurons=64, dropout_rate=0.2):
    model = Sequential([
        Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(dropout_rate),
        Dense(neurons // 2, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

# Create model
keras_classifier = KerasClassifier(build_fn=create_model, verbose=0)

# Define parameter grid
param_grid = {
    'learning_rate': [0.001, 0.01],
    'epochs': [50, 100],
    'batch_size': [16, 32, 64],
    'neurons': [32, 64, 128],
    'dropout_rate': [0.2, 0.3, 0.4]
}

# Grid search
grid = GridSearchCV(estimator=keras_classifier, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train_scaled, y_train)

# Print results
print(f"Best: {grid_result.best_score_:.4f} using {grid_result.best_params_}")
```

## Common Applications of Neural Networks for Classification

- **Image classification**: Identify objects in images
- **Text classification**: Sentiment analysis, spam filtering
- **Medical diagnosis**: Disease classification from symptoms or images
- **Customer churn prediction**: Identify customers likely to leave
- **Credit scoring**: Approve/deny loan applications
- **Fraud detection**: Identify fraudulent transactions