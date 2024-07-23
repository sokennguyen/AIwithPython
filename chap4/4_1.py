import numpy as np
import matplotlib.pyplot as plt

n = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for i in n:
    diceOne = np.random.randint(6, size=i)+1
    diceTwo = np.random.randint(6, size=i)+1
    print(diceOne)
    sum = np.add(diceOne, diceTwo)
    print(sum.size)

    h,h2 = np.histogram(sum, range(2,14))
    plt.bar(h2[:-1], h/i)
    plt.show()
