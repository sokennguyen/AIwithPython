import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

# --- 0 ---
data = pd.read_csv('data-startup.csv', skiprows=0, delimiter = ',')

# --- 1 ---
print(data.columns.values)

# --- 2 ---
hmdata = data[['R&D Spend','Administration','Marketing Spend','Profit']]
sns.heatmap(hmdata.corr().round(2).abs(), annot=True)
plt.show()

# --- 3 ---
# I would choose R&D Spend and Marketing Spend as the variables to predict the companies' profit
# I do so because they are the two variables that have the highest correlation to the profit of the companies
# Although I wouldn't put them together in a multiple linear regression model, because they are highly correlated to each other

# --- 4 --
plt.subplot(1,2,1)
plt.scatter(data['R&D Spend'], data['Profit'])
plt.xlabel('R&D Spend')
plt.ylabel('Profit')

plt.subplot(1,2,2)
plt.scatter(data['Marketing Spend'], data['Profit'])
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.show()

# --- 5 ---
# because the correlation between Marketing Spend and Profit is not as high as R&D Spend, I will only
# make a model for R&D Spend
X = data[['R&D Spend']]
y = data['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# --- 6 ---
lm = LinearRegression()
model = lm.fit(X_train, y_train)

plt.scatter(X,y)
plt.plot(X_train, model.predict(X_train), color='red')
plt.show()

# --- 7 ---
y_train_pred = model.predict(X_train)
rmseTrain = root_mean_squared_error(y_train, y_train_pred)
r2Train = r2_score(y_train, y_train_pred)

y_test_pred = model.predict(X_test)
rmseTest = root_mean_squared_error(y_test, y_test_pred)
r2Test = r2_score(y_test, y_test_pred)

print(rmseTrain, r2Train)
print(rmseTest, r2Test)
