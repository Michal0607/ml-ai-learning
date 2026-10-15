import numpy as np

X = np.array([[1,2], [2,1], [3,4], [4,3], [5,5]])
y = np.array([8, 9, 18, 19, 26])

w = np.array([0.0, 0.0])
b = 0.0
learning_rate = 0.01

def predict(X, w, b):
    y_pred = X @ w + b
    return y_pred

def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def gradients(X, y, y_pred):
    error = y_pred - y
    n = len(y)

    dw = (2 / n) * (X.T @ error)
    db = 2 * np.mean(error)

    return dw, db

def train_loop(X, y, w, b, lr, steps):
    for _ in range(steps):
        y_pred = predict(X, w, b)
        loss = mse(y, y_pred)
        dw, db = gradients(X, y, y_pred)
        w = w - lr * dw
        b = b - lr * db
    y_pred = predict(X, w, b)
    loss = mse(y, y_pred)

    return w, b, y_pred, loss

print(X.shape)
print(w.shape)

predictions = predict(X, w, b)
print(predictions)
print(predictions.shape)
error_mse = mse(y, predictions)
print(error_mse)
dw, db = gradients(X, y, predictions)
print(dw)
print(dw.shape)
print(db)

print("After 1 step of training")

w = w - learning_rate * dw
b = b - learning_rate * db

predictions = predict(X, w, b)
error_mse = mse(y, predictions)

print(predictions)
print(error_mse)

w = np.array([0.0, 0.0])
b = 0.0
steps = 10000

print(f"After trainign loop({steps}):")

w, b, y_pred, loss = train_loop(X, y, w, b, learning_rate, steps)
print("w = ", w)
print("b = ", b)
print("predictions = ", y_pred)
print("loss = ", loss)