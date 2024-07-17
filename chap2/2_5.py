import numpy as np

A = np.array([[1,2,3],[0,1,4],[5,6,0]])
a = np.linalg.inv(A)

matmulAa = np.matmul(A,a)
matmulaA = np.matmul(a,A)

print(matmulAa)
print(matmulaA)
