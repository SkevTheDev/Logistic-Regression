import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process import get_binary_data

X, Y = get_binary_data()
X, Y = shuffle(X, Y)

Xtrain = X[:-100]
Ytrain = Y[:-100]

Xtest = X[-100:]
Ytest = Y[-100:]

D = X.shape[1]
W = np.random.randn(D)
b = 0

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def forward(X, W, b):
	return sigmoid(X.dot(W) + b)

def classification_rate(Y, P):
	return np.mean(Y == P)

def cross_entropy(Y, Y_hat):
	return -np.mean(Y*np.log(Y_hat) + (1 - Y)*np.log(1 - Y_hat))

train_costs = []
test_costs = []
learning_rate = 0.001

for i in range(10000):
	Y_hat_train = forward(Xtrain, W, b)
	Y_hat_test = forward(Xtest, W, b)

	ctrain = cross_entropy(Ytrain, Y_hat_train)
	ctest = cross_entropy(Ytest, Y_hat_test)

	train_costs.append(ctrain)
	test_costs.append(ctest)

	W -= learning_rate * Xtrain.T.dot(Y_hat_train - Ytrain)
	b -= learning_rate * (Y_hat_train - Ytrain).sum()

	if i % 1000 == 0:
		print( i, ctrain, ctest)

print("Final train classification_rate:", classification_rate(Ytrain, np.round(Y_hat_train)))
print("Final test classification_rate:", classification_rate(Ytest, np.round(Y_hat_test)))

legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')

plt.legend([legend1, legend2])
plt.show()