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

poly_reg = PolynomialFeatures(degree = 2) #degree 2 means quadratic
X_poly = poly_reg.fit_transform(xpd)
print(X_poly)
