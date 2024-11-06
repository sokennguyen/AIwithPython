import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv('dataIris.csv', delimiter=',')
print(df.columns.values)

#values put the df into arrays
X = df.iloc[:,0:4].values
y = df.iloc[:,4].values
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=5)

classifier = KNeighborsClassifier(n_neighbors=5)
model = classifier.fit(X_train,y_train)
#since predictions should be done with test data anyways, y_train_test wouldn't be needed
y_pred = model.predict(X_test)

metrics.ConfusionMatrixDisplay.from_estimator(model,X_test,y_test)
plt.show()

#classification_report report on the classification models, not regression models.
#classification_report params are actual y and pred y, not any model
print(classification_report(y_test,y_pred))

error = []
for k in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    error.append(np.mean(y_pred != y_test))
    #returns an array of bool
    # -- print(y_pred != y_test) --
    #mean of a bool array is TrueCount/FalseCount
    #this calculate the True rate in that array
    # -- print((np.mean(y_pred != y_test))) --

plt.plot(range(1, 20), error, marker='o', markersize=10)
plt.xlabel('k')
plt.ylabel('Mean Error')
plt.show()

#k=7.6 and 16->17.7 provide the lowest error
