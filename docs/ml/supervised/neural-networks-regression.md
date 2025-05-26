# Neural Networks for Regression

Neural Networks for Regression are powerful supervised learning models that learn to predict continuous target variables by approximating complex, non-linear relationships between features and outputs.

## How Neural Networks Perform Regression

Neural networks for regression typically:

1. **Process input features** through multiple hidden layers
2. **Transform data** using weights, biases, and activation functions
3. **Output continuous values** from a linear output layer (no activation function)

Unlike classification networks that output probabilities, regression networks directly predict numerical values.

## Neural Network Architecture for Regression

![Neural Network Regression Architecture](https://i.imgur.com/cRLLbIJ.png)

- **Input Layer**: One neuron per feature in your dataset
- **Hidden Layers**: Multiple layers with varying numbers of neurons that use non-linear activation functions
- **Output Layer**: One or more neurons with no activation function (linear output) for continuous prediction

## Key Components for Regression

1. **Loss Functions**:
   - Mean Squared Error (MSE): Average of squared differences between predictions and targets
   - Mean Absolute Error (MAE): Average of absolute differences between predictions and targets
   - Huber Loss: Combines MSE and MAE, less sensitive to outliers

2. **Evaluation Metrics**:
   - MSE / RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - R² (Coefficient of determination)
   - Explained variance

3. **Output Layer Design**:
   - For single target regression: `Dense(1)` (no activation)
   - For multiple target regression: `Dense(n)` (no activation)

## Implementation: Single-output Regression Example

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Generate a sample regression dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scale the target
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Create a neural network for regression
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)  # No activation for regression output
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='mean_squared_error',
              metrics=['mae'])

# Display model architecture
model.summary()

# Train the model
history = model.fit(X_train_scaled, y_train_scaled, 
                    epochs=50, 
                    batch_size=32, 
                    validation_split=0.2, 
                    verbose=1)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.tight_layout()
plt.show()

# Make predictions
y_pred_scaled = model.predict(X_test_scaled).flatten()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")

# Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Neural Network Regression: Predicted vs Actual')
plt.show()

# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
```

## Implementation: Multi-output Regression Example

```python
# Generate multi-output regression dataset
X_multi, y_multi = make_regression(n_samples=1000, n_features=20, n_targets=3, noise=0.1, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.3, random_state=42)

# Scale the features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scale the targets
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Create a neural network for multi-output regression
model_multi = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(3)  # Three outputs, no activation function
])

# Compile the model
model_multi.compile(optimizer=Adam(learning_rate=0.001), 
                    loss='mean_squared_error', 
                    metrics=['mae'])

# Display model architecture
model_multi.summary()

# Train the model
history = model_multi.fit(X_train_scaled, y_train_scaled, 
                          epochs=50, 
                          batch_size=32, 
                          validation_split=0.2, 
                          verbose=1)

# Make predictions
y_pred_scaled = model_multi.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

# Evaluate individually for each output
for i in range(y_test.shape[1]):
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    print(f"Output {i+1} R-squared: {r2:.4f}")
```

## Non-linear Regression with Deep Learning

Neural networks excel at capturing non-linear relationships that traditional regression models (like linear regression) cannot:

1. **Complex Interactions**: Can model intricate interactions between features
2. **Feature Learning**: Automatically learn useful representations from raw data
3. **Flexible Modeling**: Can approximate virtually any continuous function

## Time Series Regression with Neural Networks

For sequential data like time series:

```python
from tensorflow.keras.layers import LSTM

# Create a sequence regression model with LSTM
model_time_series = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, features)),
    Dropout(0.2),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1)  # Predict next value in sequence
])

model_time_series.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

## Handling Outliers in Regression

Neural networks can be sensitive to outliers. Common strategies:

1. **Robust Loss Functions**: Use Huber loss or log-cosh loss
   ```python
   model.compile(optimizer=Adam(learning_rate=0.001), 
                 loss=tf.keras.losses.Huber(),
                 metrics=['mae'])
   ```

2. **Data Preprocessing**: Remove or cap outliers before training
   ```python
   # Identify and cap outliers
   q1, q3 = np.percentile(y_train, [25, 75])
   iqr = q3 - q1
   lower_bound = q1 - (1.5 * iqr)
   upper_bound = q3 + (1.5 * iqr)
   
   y_train_capped = np.clip(y_train, lower_bound, upper_bound)
   ```

3. **Quantile Regression**: Predict different quantiles instead of the mean
   ```python
   def quantile_loss(q, y_true, y_pred):
       error = y_true - y_pred
       return K.mean(K.maximum(q * error, (q - 1) * error), axis=-1)
       
   model.compile(optimizer='adam', 
                 loss=lambda y_true, y_pred: quantile_loss(0.5, y_true, y_pred),  # median
                 metrics=['mae'])
   ```

## Hyperparameter Tuning for Regression

Key hyperparameters to tune:

- **Network architecture**: Number of layers and neurons
- **Learning rate**: Typically between 0.1 and 0.0001
- **Batch size**: Common values: 16, 32, 64, 128
- **Activation functions**: ReLU, LeakyReLU, ELU
- **Regularization**: L1, L2, or both (ElasticNet)
- **Dropout rate**: Typically between 0.1 and 0.5

```python
# Example of hyperparameter tuning using RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

def create_model(learning_rate=0.001, neurons=64, dropout_rate=0.2):
    model = Sequential([
        Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(dropout_rate),
        Dense(neurons // 2, activation='relu'),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                 loss='mean_squared_error',
                 metrics=['mae'])
    return model

# Create model
keras_regressor = KerasRegressor(build_fn=create_model, verbose=0)

# Define parameter distribution
param_dist = {
    'learning_rate': uniform(0.0001, 0.01),
    'epochs': randint(30, 100),
    'batch_size': randint(16, 128),
    'neurons': randint(16, 128),
    'dropout_rate': uniform(0.1, 0.4)
}

# Random search
random_search = RandomizedSearchCV(estimator=keras_regressor, 
                                  param_distributions=param_dist,
                                  n_iter=20, 
                                  cv=3,
                                  scoring='neg_mean_squared_error')
random_search.fit(X_train_scaled, y_train_scaled)

# Print results
print(f"Best: {-random_search.best_score_:.4f} MSE using {random_search.best_params_}")
```

## Common Applications of Neural Networks for Regression

- **House price prediction**: Predicting property values based on features
- **Stock market forecasting**: Predicting future stock prices 
- **Energy consumption prediction**: Forecasting electricity or gas usage
- **Sales forecasting**: Predicting future sales volumes
- **Medical outcome prediction**: Estimating patient recovery time or treatment effectiveness
- **Environmental modeling**: Predicting pollution levels or weather patterns