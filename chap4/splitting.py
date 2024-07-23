import pandas as pd
import numpy as np
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('data_split.csv', skiprows=0, delimiter=',')

X = data[['CGPA']]
y = data[['Chance of Admit ']]

#data splitting for test and train
#if the train size is big, then we don't know for sure if the model is good or not
#but if the test size is big and training size is small, we can be sure that our model is shit.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test, color='red')
plt.legend(['train','test'])
plt.xlabel('CGPA')
plt.ylabel('Chance of Admit')
plt.title('Dataset splitting')

#linreg is a transformer
linreg = linear_model.LinearRegression()
#model is a MODEL, created by fitting data into our transformer
model = linreg.fit(X_train, y_train)

plt.scatter(X,y, color='blue')
plt.plot(X_test, model.predict(X_test), color='red')
plt.title('Predicted')
plt.show()
