import numpy as np

data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)
height = data[:,1]
weight = data[:,2]

height = height * 2.54
weight = weight * 0.453592

meanHeight = np.mean(height)
meanWeight = np.mean(weight)
medianHeight = np.median(height)
medianWeight = np.median(weight)
stdHeight = np.std(height)
stdWeight = np.std(weight)
varHeight = np.var(height)
varWeight = np.var(weight)
