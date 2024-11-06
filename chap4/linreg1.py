import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data= pd.read_csv('data.csv',skiprows=0,names=['x','y'])

xpd = data['x']
ypd = data['y']
print(data.head())

xbar = np.mean(xpd)
ybar = np.mean(ypd)
term1 = np.sum(xpd*ypd)
term2 = np.sum(xpd**2)
n= xpd.size
b = (term1 - n*xbar*ybar)/(term2 - n*xbar**xbar)
a = ybar - b*xbar

x = np.linspace(0,2,50)
y = a + b*x
plt.plot(x,y, color = 'black')
plt.scatter(xpd,ypd)
plt.scatter(xbar,ybar, color = 'red')
#plt.show()

xval = 0.5
yval = a+b*xval

xValArr = np.array([0.5,1.0,1.5])
yValArr = a+b*xValArr
print(yValArr)
