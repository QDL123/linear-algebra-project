# qr decomposition to do least square
from numpy import array
from numpy.linalg import inv
from numpy.linalg import qr
from matplotlib import pyplot
import pandas as pd
data = pd.read_csv('waterData6.csv')
X = data['T_degC'].values
Y = data['Salnty'].values

#X, y = data[:,0], data[:,1]
X = X.reshape((len(X), 1))
# QR decomposition
print(X)
print("That was x")
X = array(X)
print(X)
print("Updated X above")
Q, R = qr(X)
#print(Q)
print("and now b")
#print(R)
b = inv(R).dot(Q.T).dot(Y)
At = inv(R).dot(Q.T)
print(At)
print("that was At")
AtSmall = At/1000
print(AtSmall)
print("that was small")
#print("this is Y below")
#print(Y)
AtSmallb = AtSmall.dot(Y)
print("and AtSmallb")
print(AtSmallb)
#print("and b")
#print(b)
# predict using coefficients
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
