import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

# --- 1 ---
df = pd.read_csv('dataAuto.csv', delimiter=',')
print(df.columns.values)

# --- 2 ---
X = pd.DataFrame(df[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year']])
y = df['mpg']

# --- 3 ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# --- 4, 5, 6 ---
ridgeAlphas = np.linspace(0, 350, 50)
ridgeRes = []
for alp in ridgeAlphas:
    rr = Ridge(alpha=alp)
    rr.fit(X_train, y_train)
    predR2 = r2_score(y_test, rr.predict(X_test))
    ridgeRes.append(predR2)

lassoAlphas = np.linspace(0, 10, 50)
lassoRes = []
for alp in lassoAlphas:
    lr = Lasso(alpha=alp)
    lr.fit(X_train, y_train)
    #tried using lr.score but got warned to use r2_score instead
    predR2 = r2_score(y_test, lr.predict(X_test))
    lassoRes.append(predR2)



plt.subplot(1, 2, 1)
plt.plot(ridgeAlphas, ridgeRes, label='Ridge')
plt.xlabel('alpha')
plt.ylabel('R2')

plt.subplot(1, 2, 2)
plt.plot(lassoAlphas, lassoRes, label='Lasso')
plt.xlabel('alpha')
plt.ylabel('R2')
plt.show()

# --- 7 ---
ridgeAlphas = np.linspace(91.8, 92.2, 50)
ridgeRes = []
for alp in ridgeAlphas:
    rr = Ridge(alpha=alp)
    rr.fit(X_train, y_train)
    predR2 = r2_score(y_test, rr.predict(X_test))
    ridgeRes.append(predR2)


lassoAlphas = np.linspace(0.2, 0.5, 50)
lassoRes = []
for alp in lassoAlphas:
    lr = Lasso(alpha=alp)
    lr.fit(X_train, y_train)
    predR2 = r2_score(y_test, lr.predict(X_test))
    lassoRes.append(predR2)

plt.subplot(1, 2, 1)
plt.plot(ridgeAlphas, ridgeRes)
plt.xlabel('alpha')
plt.ylabel('R2')

plt.subplot(1, 2, 2)
plt.plot(lassoAlphas, lassoRes)
plt.xlabel('alpha')
plt.ylabel('R2')
plt.show()

# 92.07 is the best alpha for Ridge
# 0.3 is the best alpha for Lasso
