import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('dataquad.csv', skiprows = 0, names = ['x','y'])

xpd = np.array(data[['x']])
ypd = np.array(data[['y']])
xpd = xpd.reshape(-1,1)
ypd = ypd.reshape(-1,1)

#poly_reg here is a polynomial sklearn object, which have polynomial functions
poly_reg = PolynomialFeatures(degree = 2) #degree 2 means quadratic
#in polynomial regression, an array of powers from 0 -> 1 of x is needed, this line creates that array
X_poly = poly_reg.fit_transform(xpd)
#I dont know why do we need to fit X_poly into a linear regression model
pol_reg = LinearRegression()
#poly_reg here is a transformer, which was fitted with X_poly and ypd
pol_reg.fit(X_poly,ypd)

plt.scatter(xpd,ypd, color='red')
xval = np.linspace(-1,1,100).reshape(-1,1) #why reshape? because the predict function expects a 2D array and xval is a 1D array
#this line predicts y based on the transformed xval
#the transformation is done with poly_reg, which is the polynomial transformer
#the prediction is done with pol_reg, the linear regression transformer
plt.plot(xval, pol_reg.predict(poly_reg.fit_transform(xval)), color='blue')
plt.show()
print(pol_reg.coef_)
print("c=", pol_reg.intercept_)
