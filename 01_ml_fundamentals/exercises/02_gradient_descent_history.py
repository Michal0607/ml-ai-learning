import numpy as np

X = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9 , 11])

w = 0
b = 0
learning_rate = 0.01

def predict(X, w, b):
    return w * X + b

def mse(y, y_pred):
    return np.mean((y-y_pred) ** 2)

def gradients(X, y, y_pred):
    dw = 2 * (np.mean((y_pred - y) * X))
    db = 2 * (np.mean(y_pred - y))
    return dw, db

print("Parameters before training:")

print("X: ", X)
print("y: ", y)
print("w: ", w)
print("b: ", b)

def train_loop(X, y, w, b):
    loss_history = []
    for _ in range(100):
        y_pred = predict(X, w ,b)
        mse_error = mse(y, y_pred)
        loss_history.append(float(mse_error))
        dw, db = gradients(X ,y ,y_pred)
        w = w - learning_rate * dw
        b = b - learning_rate * db
    y_pred = predict(X, w ,b)
    mse_error = mse(y, y_pred)
    loss_history.append(float(mse_error))

    return y_pred, loss_history, w, b

y_pred, loss_history, w, b = train_loop(X, y, w ,b)

print("Parameters after training:")

print("X: ", X)
print("y: ", y)
print("w: ", w)
print("b: ", b)
print("loss history first 5 steps:", [round (x,4) for x in loss_history[:5]])
print("loss history last 5 steps:", [round (x,4) for x in loss_history[-5:]])