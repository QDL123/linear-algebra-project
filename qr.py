# qr decomposition to do least square
from numpy import array
from numpy.linalg import inv
from numpy.linalg import qr
from matplotlib import pyplot
import panda as pd
data = read_csv('file_name.csv')
X, y = data[:,0], data[:,1]
X = X.reshape((len(X), 1))
# QR decomposition
Q, R = qr(X)
b = inv(R).dot(Q.T).dot(y)
print(b)
# predict using coefficients
yhat = X.dot(b)
# plot data and predictions
pyplot.scatter(X, y)
pyplot.plot(X, yhat, color='red')
pyplot.show()
