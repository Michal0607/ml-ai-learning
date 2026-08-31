import numpy as np

X = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])

w = 0
b = 0
learning_rate = 0.01

def predict(X, w ,b):
    return w * X + b

def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def gradients(X, y, y_pred):
    dw = 2 * (np.mean((y_pred - y) * X))
    db = 2 * (np.mean(y_pred - y))
    return dw, db

predictions = predict(X, w, b)
dw, db = gradients(X, y, predictions)

print("dw:" , dw, "db:" , db)
print("True values:", y)
print("Predictions:", predictions)
print("MSE: ", mse(y, predictions))

w = w - learning_rate * dw
b = b - learning_rate * db

predictions = predict(X, w, b)
dw, db = gradients(X, y, predictions)

print("After 1 step")
print("w:", w, "b:", b)
print("dw:" , dw, "db:" , db)
print("True values:", y)
print("Predictions:", predictions)
print("MSE: ", mse(y, predictions))

MSE_error = mse(y, predictions)

def train_loop(X, y, w, b):
    for number in range(100):
        predictions = predict(X, w, b)
        loss = mse(y, predictions)
        dw, db = gradients(X, y, predictions)
        w = w - learning_rate * dw
        b = b - learning_rate * db
    predictions = predict(X, w, b)
    loss = mse(y, predictions)
    return predictions, w, b, loss

predictions, w, b, loss = train_loop(X, y, w, b)

print("After training loop")
print("w:", w, "b:", b)
print("dw:" , dw, "db:" , db)
print("True values:", y)
print("Predictions:", predictions)
print("MSE: ", loss)