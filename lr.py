import numpy as np
data = np.loadtxt("data.csv", delimiter=",", skiprows=1)
np.random.shuffle(data)
split = int(0.7 * len(data))
train = data[:split]
test = data[split:]
X_train = train[:, :3] # training data
y_train = train[:, 3].astype(int)
X_test = test[:, :3]
y_test = test[:, 3].astype(int)

w = np.array([0.0,0.0,0.0])
b = 0

def sigmoid(z):
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z)) # if z<0 then just return this value, does not overflow as much
    )
def calc_gradient(X,y,w,b):
    z = X@w + b
    p = sigmoid(z)
    m = X.shape[0]
    dw = (1/m)*(X.T @ (p-y))
    db = (1/m)*(np.sum(p-y))
    return dw, db

i=0
alpha=0.001
while i<10000:
    dw,db = calc_gradient(X_train,y_train,w,b)
    w = w - alpha*dw
    b = b - alpha*db
    i += 1

print("weights:", w)
z = X_test @ w + b
p = sigmoid(z)
pred = (p >= 0.5).astype(int)
accuracy = np.mean(pred == y_test)
print("Test accuracy:", accuracy)