import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,10,100)
y = 2*x+1
plt.plot(x, y, linestyle="--")
y = 2*x+2
plt.plot(x, y)
y = 2*x+3
plt.plot(x, y, linestyle=":")


plt.title("Title")
plt.xlabel("x")
plt.ylabel("y")


plt.show()
