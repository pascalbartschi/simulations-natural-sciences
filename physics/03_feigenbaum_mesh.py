import numpy as np
import matplotlib.pyplot as plt

length = 100
aListLen = length
xListLen = length

aList = np.linspace(0.0, 4, aListLen)
xList = np.linspace(0.0, 1, xListLen)

A, X = np.meshgrid(aList, xList)


# print(A)
# print(X)


def LogEq(a, x):
    # make 1000 iterations
    for n in range(1000):
        x = a * x * (1 - x)
    return x


feigenbaum = LogEq(A, X)
# print(feigenbaum)

# nplot = 500  # do not plot the initial transition

plt.figure()

for i in range(len(aList)):
    plt.scatter(aList[i] * np.ones_like(feigenbaum[i][nplot:]),
                feigenbaum[i][nplot:], s=0.01, c='k')

plt.show()
