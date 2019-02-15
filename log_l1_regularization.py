import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

N = 50
D = 50

X = (np.random.random((N, D)) - 0.5)*10

true_w = np.array([1, 0.5, -0.5] + [0]*(D-3))

Y = np.round(sigmoid(X.dot(true_w) + np.random.randn(N)*0.5))

errors = []
w = np.random.rand(D) /np.sqrt(D)

learning_rate = 0.001

l1 = 7.0
for t in range(5000):
	Y_hat = sigmoid(X.dot(w))
	delta = Y_hat - Y
	w -= learning_rate * (X.T.dot(delta) + l1*np.sign(w))

	error = -np.mean(Y*np.log(Y_hat) + (1 - Y)*np.log(1 - Y_hat)) + np.mean(l1*np.abs(w))
	errors.append(error)

plt.plot(errors)
plt.show()

plt.plot(true_w, label='true w')
plt.plot(w, label='w map')
plt.legend()
plt.show()