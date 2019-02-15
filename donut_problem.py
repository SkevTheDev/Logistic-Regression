import numpy as np
import matplotlib.pyplot as plt

N = 1000
D = 2

R_inner = 5
R_outer = 10

R1 = np.random.randn(int(N/2)) + R_inner
theta = 2*np.pi*np.random.random(int(N/2))
X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

R2 = np.random.randn(int(N/2)) + R_outer
theta = 2*np.pi*np.random.random(int(N/2))
X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

X = np.concatenate([X_inner, X_outer])
Y = np.array([0]*(int(N/2)) + [1]*(int(N/2)))

plt.scatter(X[:,0], X[:,1], c=Y)
plt.show()

ones = np.array([[1]*N]).T

r = np.zeros((N,1))
for i in range(N):
	r[i] = np.sqrt(X[i,:].dot(X[i,:]))

Xb = np.concatenate((ones, r, X), axis=1)

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

learning_rate = 0.0001
l2 = 0.01
errors = []
for i in range(5000):
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