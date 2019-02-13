import numpy as np
import matplotlib.pyplot  as plt

N = 100
D = 2

X = np.random.randn(N, D)

#center first 50 points at (-2,-2)
X[:50, :] = X[:50,:] - 2*np.ones((50, D))

#center last 50 points at (2,2)
X[50:, :] = X[50:,:] + 2*np.ones((50, D))

# labels first 50 are 0, last 50 are 1
Y = np.array([0]*50 + [1]*50)

ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis=1)

w = np.random.randn(D+1)

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

print(cross_entropy(Y, Y_hat))

#closed form bayes classifier solution 
#0 is bias
w = np.array([0,4,4])

z = Xb.dot(w)
Y_hat = sigmoid(z)

print(cross_entropy(Y, Y_hat))

plt.scatter(X[:,0], X[:, 1], c=Y, s=100, alpha=0.5)

x_axis = np.linspace(-6,6, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)

plt.show()