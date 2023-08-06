import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import curve_fit



def simulation(gridrange,c_init, method, D, dx, dt, steps):

    k = D * (dt / dx)

    grid = np.zeros(gridrange)
    grid[int(len(grid)/2)] = c_init
    
    x0 = np.arange(0, gridrange)

    gridlist = [grid]
    paramlist = [[np.nan, np.nan]]

    for i in range(steps):
        grid = method(grid, D, dt, dx)
        # popt, pcov = curve_fit(gauss, x0, grid)
        
        gridlist.append(grid)
        # paramlist.append([max(grid), np.std(grid)])
        
    

    return np.array(gridlist), np.array(paramlist)


def method(grid, D, dt, dx):
    
    return (np.roll(grid, 1) + np.roll(grid, -1) - 2 * grid) * D * dt / dx**2


def gauss(x, a, mu, sigma):
    
    return a * np.exp(-(x-mu) ** 2 / (2 * sigma ** 2))
    

def euler_method(grid, D, dt, dx):
    
    delta = method(grid, D, dt, dx)   
    
    return grid + delta


def rk2_method(grid, D, dt, dx):
    
    delta_mid = method(grid, D, dt/2, dx)
    grid_mid = grid + delta_mid
    delta = method(grid_mid, D, dt, dx)
    
    return grid + delta
    



D = 0.4
dt = 0.01
dx = 5
steps = 100
gridrange = 100
c_init = 1

res, pars = simulation(gridrange = gridrange,
                 c_init = c_init,
                 method = euler_method,
                 D = D,
                 dt = dt,
                 dx = dx,
                 steps = steps)

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes()
line, = ax.plot([], [])
# xdata, ydata = [], []


# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,


# animation function.  This is called sequentially
def animate(i):
    ax.clear()
    x = np.arange(0, gridrange * dx, dx)
    y = res[i, :]
    y0 = gauss(x,1,*pars[i,:])
    line.set_data(x, y)
    ax.plot(x, y)
    ax.plot(x, y0)
    # print(i)
    return line,


# call the animator.  blit=True means only re-draw the parts that have changed.
anim = FuncAnimation(fig, animate, frames=np.arange(steps), interval=20)

anim.save("animation.mp4", fps=30, dpi=300, extra_args=['-vcodec', 'libx264'])

# plt.plot(res[75, :])
# plt.show()