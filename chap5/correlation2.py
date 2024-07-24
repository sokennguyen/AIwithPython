import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('./winedata/winequality-white.csv', delimiter=';')
print(df.head())

sns.heatmap(data=df.corr().round(2).abs(), annot=True)

plt.clf()
plt.subplot(1,2,1)
plt.scatter(df['residual sugar'],df['density'])
plt.xlabel("residual sugar")
plt.ylabel("density")

plt.subplot(1,2,2)
plt.scatter(df['alcohol'],df['density'])
plt.xlabel("alcohol")
plt.ylabel("density")
plt.show()

X = pd.DataFrame(df[['residual sugar','alcohol']], columns=['residual sugar','alcohol'])
y = df['density']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

lm = LinearRegression()
model = lm.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
rmse = root_mean_squared_error(y_train, y_train_pred)
r2 = r2_score(y_train, y_train_pred)

y_test_pred = model.predict(X_test)
rmsetest = root_mean_squared_error(y_test, y_test_pred)
r2test = r2_score(y_test, y_test_pred)

print(rmse, r2)
print(rmsetest,r2test)
