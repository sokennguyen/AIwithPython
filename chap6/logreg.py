import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv('dataLogreg.csv', delimiter=',')
print(df.head())

#iloc[:, _] output everything in the column
X = df.iloc[:, 0:2] #column 2 is excluded
y = df.iloc[:, -1] #last column is selected

admit_yes = df.loc[y == 1] #df.loc searches at "y", which is the last column
admit_no = df.loc[y == 0]

plt.scatter(admit_no.iloc[:,0], admit_no.iloc[:,1], label='not admitted')
plt.scatter(admit_yes.iloc[:,0], admit_yes.iloc[:,1], label='admitted')
plt.xlabel('exam 1')
plt.xlabel('exam 2')
plt.legend()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
lr = LogisticRegression()
model = lr.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_test_pred)
print(cnf_matrix)

#this is a graph made by sklearn
metrics.ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.show()

#have to move to numpy to make calculations
y_test2 = y_test.to_numpy()
#these are the conditions in y_test2
idx1 = np.logical_and(y_test_pred == 1, y_test2 == 1)
#logical_and output boolean
print('idx1: ', idx1)
idx2 = np.logical_and(y_test_pred == 1, y_test2 == 0)
idx3 = np.logical_and(y_test_pred == 0, y_test2 == 0)
idx4 = np.logical_and(y_test_pred == 0, y_test2 == 1)
#locate the rows with the conditions
X1 = X_test.loc[idx1] #look at X_test, locate the rows where the predicted val is 1 and the actual val is 1
#this is hard to understand because X_test don't have the data of y_test_pred and y_test2 inside of it.
#EX: X_test.loc[true]?
print('X1: ', X1)
X2 = X_test.loc[idx2]
X3 = X_test.loc[idx3]
X4 = X_test.loc[idx4]

plt.scatter(X1.iloc[:,0],X1.iloc[:,1],label="pred yes correct",marker="+",color="blue")
plt.scatter(X2.iloc[:,0],X2.iloc[:,1],label="pred yes incorrect",marker="o",color="blue")
plt.scatter(X3.iloc[:,0],X3.iloc[:,1],label="pred no correct",marker="+",color="red")
plt.scatter(X4.iloc[:,0],X4.iloc[:,1],label="pred yes incorrect",marker="o",color="red")

plt.xlabel("exam1")
plt.ylabel("exam2")
plt.legend()
plt.title("Predicted")
plt.show()
