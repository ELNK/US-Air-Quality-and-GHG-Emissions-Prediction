import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

def train_test_split(data, train_size=0.8):
    idx = np.random.permutation(len(data))
    split = int(train_size * len(data))
    return data.iloc[idx[:split]], data.iloc[idx[split:]]

def rmse(pred, actual):
    return np.sqrt(np.mean((actual - pred)**2))

def train_model(X_train, y_train, model_type='linear'):
    if model_type == 'linear':
        model = LinearRegression()
    else:
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    error = rmse(preds, y_test)
    return preds, error

def plot_residuals(y_true, y_pred, weights=None):
    plt.figure(figsize=(12, 6))
    plt.scatter(y_true, y_pred - y_true, s=5)
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('Residuals')
    plt.show()
