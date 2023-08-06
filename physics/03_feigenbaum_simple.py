import numpy as np
import matplotlib.pyplot as plt


def LogEq(a, xList):
    xinfList = []
    for x in xList:
        xn = x
        for n in range(1000):
            xn = a * xn * (1 - xn)
        xinfList.append(xn)
    return xinfList


# define

length = int(1e3)
aListLen = length
xListLen = length

aList = np.linspace(0.0, 4, aListLen)
xList = np.linspace(0.0, 1, xListLen)

plt.figure()

for a in aList:
    xinfList = LogEq(a, xList)
    del (xinfList[0], xinfList[-1])
    plt.scatter(a * np.ones(length - 2), xinfList, s=0.01, c="k")

plt.xlim(1, 4)
plt.savefig("feigenbaum.png")  # pdf file becomes huge, almost unopable by most applications
plt.show()
