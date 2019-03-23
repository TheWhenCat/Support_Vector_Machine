#imports
import numpy as np
#plotting
from matplotlib import pyplot as plt

#Input Data - [x,y,bias]
x = np.array([
    [-2,-4,-1],
    [4,1,-1],
    [1,6,-1],
    [2,4,-1],
    [6,2,-1]
])

#Associtaed Output
y = np.array([-1,-1,1,1,1])

for d, sample in enumerate(x):
    #print(d, sample)
    if d<2:
        plt.scatter(sample[0], sample[1], s = 120, marker='_', linewidths=2)
    else:
        plt.scatter(sample[0], sample[1], s = 120, marker = '+', linewidths=2)

#random line (hyperplane or separator in this case)
#plt.plot([-2,6], [6,0.5])


def svm_sgd_plot(X,Y):
    w = np.zeros(len(X[0]))

    eta = 1

    epochs = 100000

    errors = []

    for epoch in range(1, epochs):
        error = 0
        for i, x in enumerate(X):
            # misclassification
            if (Y[i] * np.dot(X[i], w)) < 1:
                # misclassified update for ours weights
                w = w + eta * ((X[i] * Y[i]) + (-2 * (1 / epoch) * w))
                error = 1
            else:
                # correct classification, update our weights
                w = w + eta * (-2 * (1 / epoch) * w)
        errors.append(error)



    return w

w = svm_sgd_plot(x, y)

for d, sample in enumerate(x):
    #print(d, sample)
    if d<2:
        plt.scatter(sample[0], sample[1], s = 120, marker='_', linewidths=2, color='green')
    else:
        plt.scatter(sample[0], sample[1], s = 120, marker = '+', linewidths=2, color='green')

#Test Samples
plt.scatter(2,2,s = 120, marker='_', linewidths=2, color='yellow')
plt.scatter(4,3,s = 120, marker='+', linewidths=2, color='blue')

x2 = [w[0], w[1], -w[1], w[0]]
x3 = [w[0], w[1], w[1], -w[0]]
x2x3 = np.array([x2,x3])
X,Y,U,V = zip(*x2x3)
ax = plt.gca()
ax.quiver(X,Y,U,V, scale=1, color='blue')
plt.show()

