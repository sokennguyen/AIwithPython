import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data = pd.read_csv('dataWH.csv', skiprows=0, delimiter=',')

h = data[['Height']]
w = data[['Weight']]
plt.scatter(h,w)

linreg = LinearRegression()
#in linear reg models, first param is x, which is the independent value, which also means the input
#the second param is the output, a.k.a dependent value
model = linreg.fit(h,w)

print("R2=",metrics.r2_score(w,model.predict(h)))
print("RMSE=",metrics.root_mean_squared_error(w,model.predict(h)))
#when doing predictions, the input is the x/independent value, and the output is the y/dependent val
plt.plot(h, model.predict(h), color='red')
plt.show()
