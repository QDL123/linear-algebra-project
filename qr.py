# qr decomposition to do least square
from numpy import array
from numpy.linalg import inv
from numpy.linalg import qr
from matplotlib import pyplot
import pandas as pd
data = pd.read_csv('waterData7.csv')
X = data['T_degC'].values
Y = data['Salnty'].values

#X, y = data[:,0], data[:,1]
X = X.reshape((len(X), 1))
X = array(X)
# QR decomposition
Q, R = qr(X)
At = inv(R).dot(Q.T)
b = At.dot(Y)
print(b)
yhat = X.dot(b)
# plot data and predictions
pyplot.scatter(X, Y)
pyplot.plot(X, yhat, color='red')
# x-axis label
pyplot.xlabel('Temperature')
#y-axis label
pyplot.ylabel('Water Salinity')
pyplot.legend()
pyplot.show()
print(yhat)
pyplot.show()
