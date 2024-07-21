import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('dataquad.csv', skiprows=0, names=['x','y'])

xpd = np.array(data[['x']])
ypd = np.array(data[['y']])

xbar = np.mean(xpd)
ybar = np.mean(ypd)

n = xpd.size
xpd = xpd.reshape((n,))
ypd = ypd.reshape((n,))

Sxx = np.sum(xpd**2)-n*xbar**2
Sxy = np.dot(xpd,ypd)-n*xbar*ybar
Sxx2 = np.sum(xpd**3)-xbar*np.sum(xpd**2)
Sx2y = np.sum(xpd**2*(ypd))-ybar*np.sum(xpd**2)
Sx2x2 = np.sum(xpd**4)-(np.sum(xpd**2)**2)/n

a = (Sx2y*Sxx-Sxy*Sxx2)/(Sxx*Sx2x2-Sxx2**2)
b = (Sxy*Sx2x2-Sx2y*Sxx2)/(Sxx*Sx2x2-Sxx2**2)
c = ybar-b*xbar-a*np.sum(xpd**2)/n

x = np.linspace(-1,1,100)
y = a*x**2 + b*x + c

yhat = a*xpd**2 + b*xpd +c
R2 = 1-np.sum((ypd-yhat)**2)/np.sum((ypd-ybar)**2)
print(R2)

RMSE = np.sqrt(np.sum((ypd-yhat)**2)/n)
print(RMSE)

plt.plot(x,y, color = 'black')
plt.scatter(xpd,ypd)
plt.show()
