import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# --- 1 ---
df = pd.read_csv('dataBank.csv', delimiter=';', skiprows=0)
print(df.columns.values)

# --- 2 ---
df2 = df[['y','job','marital','default','housing','poutcome']]
print(df2.head())

# --- 3 ---
df3 = pd.get_dummies(df2,columns=['job','marital','default','housing','poutcome'])
print(df3.head())

# --- 4 ---
sns.heatmap(data=df3.drop(columns=['y']).corr().round(1).abs(), annot=True)
plt.show()
# little to no correlation between the variables
# besides poutcome_unknown lead to  poutcome_failure > poutcome_other > poutcome_success

# --- 5 ---
y = df3['y']
X = df3.drop(columns=['y'])
print(X.head())

# --- 6 ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# --- 7 ---
lr = LogisticRegression()
linearModel = lr.fit(X_train, y_train)
lrPred = linearModel.predict(X_test)

# --- 8 ---
cnf_matrix = metrics.confusion_matrix(y_test, lrPred)
print(cnf_matrix)
print(metrics.accuracy_score(y_test, lrPred))

# --- 9_7 ---
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

# --- 9_8 ---
cnf_matrix = metrics.confusion_matrix(y_test, knn_pred)
print(cnf_matrix)
print(metrics.accuracy_score(y_test, knn_pred))

# --- 10 ---
# The logistic regression model is better than the KNN model
