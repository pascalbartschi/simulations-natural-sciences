# libraries

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# def func1(x):
#     return x** x - 100
#
#
# def func2(x):
#     x ** 3 + x ** 2...
#
#
# def bisection(f, a, b):
#     if f(c) > 0:
#
#
# x0 = bisection(func1, 0, 4)
# print(x0)
#
# x0 = bisection(func2,..,..)


e = 0.5


def fKepler(E, M, e):
    return E - e * np.sin(E) - M


def fKeplerPrime(E, e):
    return 1 - e * np.cos(E)  # check if done right!


def newton(f, fPrime, M, e, Estart):
    E = Estart
    Enew = E + 1
    tol = 0.001
    while abs(E - Enew) > tol:
        dx = -f(E, M, e) / fPrime(E, e)
        Enew = E + dx
        E = Enew
    return E


def update(frame):
    global xdata, ydata
    a = 1
    M = frame
    E = newton(fKepler, fKeplerPrime, M, e, M)
    x = a * np.cos(E) - a * e
    y = a * ((1 - e ** 2) ** 0.5) * np.sin(E)

    xdata.append(x)
    ydata.append(y)
    ln.set_data(xdata, ydata)
    return ln,


def init():
    # plt.cla()
    ax.set_xlim(-2, 1)
    ax.set_ylim(-1, 1)
    return ln,


fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro', animated=True)

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2 * np.pi, 50),
                    init_func=init, blit=True, interval=100)

ani.save("Keplers_Equation.mp4", writer="ffmpeg", dpi=250)
plt.show()
