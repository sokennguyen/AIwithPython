import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

df = pd.read_csv('dataridge.csv', delimiter=',')
X = df[['x']]
y = df[['y']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# trying out all of the alpha values
for alp in [0,1,5,10,20,30,50,100,1000]:
    rr = Ridge(alpha=alp)
    rr.fit(X_train, y_train)
    plt.scatter(X_train, y_train)
    plt.plot(X_train, rr.predict(X_train),color = 'red')
    plt.title('alpha = '+str(alp))

# results fall of after alpha = 4
alphas = np.linspace(0, 4, 50)
r2Values = []
for alp in alphas:
    rr= Ridge(alpha=alp)
    rr.fit(X_train, y_train)
    r2_test = r2_score(y_test, rr.predict(X_test))
    r2Values.append(r2_test)

plt.clf()
plt.plot(alphas, r2Values)
plt.show()
