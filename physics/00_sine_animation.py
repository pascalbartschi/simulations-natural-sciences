# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 13:53:34 2018

@author: sts
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def userFunction(frame):
    return np.sin(frame)


def init():
    # plt.cla()
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1, 1)
    return ln,


def update(frame):
    global xdata, ydata
    x = frame
    y = userFunction(x)
    xdata.append(x)
    ydata.append(y)
    # xdata = [x]
    # ydata = [y]
    ln.set_data(xdata, ydata)
    return ln,


fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro', animated=True)

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2 * np.pi, 128),
                    init_func=init, blit=True, interval=50, repeat=False)

ani.save("sin.mp4", writer="ffmpeg", dpi=250)
plt.show()
