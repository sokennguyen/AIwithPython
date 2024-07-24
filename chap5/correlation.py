# intro correlation
import pandas as pd
data = pd.read_csv('dataWH.csv', usecols=[1,2], delimiter=',')
print(data.corr())

# --- p1 ---
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score

#new libs
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

data = load_diabetes(as_frame=True) #as_frame for turning it into a typical pandas dataframe
df = data.frame
print(df.head())

plt.hist(df['target'], 25)
plt.xlabel('target')

sns.heatmap(data=df.corr().round(2), annot=True)
plt.show()

plt.subplot(1,2,1)
plt.scatter(df['bmi'], df['target'])
plt.xlabel('bmi')
plt.ylabel('target')

plt.subplot(1,2,2)
plt.scatter(df['s5'], df['target'])
plt.xlabel('s5')
plt.ylabel('target')

X = pd.DataFrame(df[['bmi', 's5', 'bp', 's4']], columns=['bmi', 's5', 'bp', 's4'])
y= df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

lm = LinearRegression()
model = lm.fit(X_train, y_train)

#predicting with training data
y_train_pred = model.predict(X_train)
RMSE = root_mean_squared_error(y_train, y_train_pred)
r2 = r2_score(y_train, y_train_pred)

#predicting with testing data
y_test_pred = model.predict(X_test)
RMSEtest = root_mean_squared_error(y_test, y_test_pred)
r2test = r2_score(y_test, y_test_pred)

print(RMSE,r2)
print(RMSEtest, r2test)
