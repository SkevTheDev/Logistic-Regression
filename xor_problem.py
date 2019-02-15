import numpy as np
import matplotlib.pyplot as plt

N = 4
D = 2

X = np.array([
		[0,0],
		[0,1],
		[1,0],
		[1,1]
])

Y = np.array([0,1,1,0])

ones = np.array([[1]*N]).T

plt.scatter(X[:,0], X[:,1], c=Y)

plt.show()

xy = np.matrix(X[:,0] * X[:,1]).T

Xb = np.array(np.concatenate((ones, xy, X), axis=1))

w = np.random.rand(D+2)

z = Xb.dot(w)

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

Y_hat = sigmoid(z)

def cross_entropy(Y, Y_hat):
	E = 0
	for i in range(N):
		if Y[i] == 1:
			E -= np.log(Y_hat[i])
		else:
			E -= np.log(1 - Y_hat[i])
	return E

learning_rate = 0.1
l2 = 0.01
errors = []
for i in range(2000):
	error = -np.mean(Y*np.log(Y_hat) + (1 - Y)*np.log(1 - Y_hat))
	errors.append(error)
	if i % 100 == 0:
		print(error)
	w -= learning_rate * (Xb.T.dot(Y_hat - Y) + l2*w)
	Y_hat = sigmoid(Xb.dot(w))

plt.plot(errors)
plt.show()

print("Final w:", w)	
print("Final Classification Rate: ", 1 - np.abs(Y - np.round(Y_hat)).sum() / N)