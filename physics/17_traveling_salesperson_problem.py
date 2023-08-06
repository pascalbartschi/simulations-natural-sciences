import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import jit
import time
import progressbar

@jit(nopython=True)
def energy(x, y, tour):
    e = 0
    for i in range(len(tour)):
        e += ((x[tour[i]] - x[tour[i-1]])**2 + (y[tour[i]] - y[tour[i-1]])**2)**0.5

    return e


    # return (((x - np.roll(x, -1)) ** 2 + (y - np.roll(y, -1)) ** 2) ** 0.5).sum()

@jit(nopython=True)
def city_swap(x, y, tour, T):

    i, j = np.random.randint(low = 0, high = len(x)-1, size = 2)

    # hypothetical new configuration
    tournew = tour.copy()
    tournew[i] = tour[j]
    tournew[j] = tour[i]

    dE =  energy(x, y, tournew) - energy(x, y, tour)

    if dE < 0:
        tour = tournew
    else:
        if np.exp(-dE / T) > np.random.uniform(0, 1):
            tour = tournew

    return tour
#
# def dist(p, q):
#
#     return np.sum((q-p) ** 2)

@jit(nopython=True)
def reverse_segment(x, y, tour, T):

    i, j = np.random.randint(low = 0, high = len(x)-1, size = 2)
    if i > j:
        t = i
        i = j
        j = t

    # hypothetical new configuration
    tournew = tour.copy()
    t = np.flip(tour[i:j])
    tournew[i:j] = t # tour.copy()[j-1:i-1:-1]

    dE =  energy(x, y, tournew) - energy(x, y, tour)

    if dE < 0:
        tour = tournew
    else:
        if np.exp(-dE / T) > np.random.uniform(0, 1):
            tour = tournew

    return tour


@jit(nopython=True)
def estimate_T0():
    """
    estimates highest dE
    :return: T0 for sim
    """
    maxdE = 0
    for _ in range(100):
        i, j = np.random.randint(low=0, high=len(x) - 1, size=2)

        # hypothetical new configuration
        tournew = tour.copy()
        tournew[i] = tour[j]
        tournew[j] = tour[i]

        dE = energy(x, y, tour) - energy(x, y, tournew)

        if dE > maxdE:
            maxdE = dE

    return maxdE

# import city nodes
x, y = np.loadtxt("ch130.tsp", delimiter=' ', comments="EOF",
                  skiprows=6, usecols=(1, 2), unpack=True)

tour = np.arange(len(x),dtype = np.int64)


# import optimal solution / global minimum
opttour = np.loadtxt("ch130.opt.tour", delimiter=' ', comments="-1",
                  dtype=int, skiprows=5, usecols=(0), unpack=True)

T0 = estimate_T0()
N_perstep = 20 * len(x) ** 2
Tend = 0.1


@jit(nopython=True)
def sim(x, y, tour, T0, N_perstep, Tend):
    T = T0
    Ts = [T]
    store_tour = [tour]
    while T > Tend:
        for _ in range(N_perstep):
            if np.random.uniform(0,1) < 0.5:
                tour = city_swap(x, y, tour, T)
            else:
                tour = reverse_segment(x, y, tour, T)

        store_tour.append(tour.copy())
        Ts.append(T)
        T *= 0.98 # reduce T of system

    return Ts, store_tour



start = time.time()
#store_S, steps , i_eq = sim()
Ts, store_tour = sim(x, y, tour, T0, N_perstep, Tend)
store_tour.append(store_tour[-1].copy()) # due to plotting last frame issues in animate
# store_S = np.array(store_S)
stop = time.time()
print(f"Compiling simulation took {round(stop - start, 2)} seconds.")
fig, ax = plt.subplots(1, 1)
bar = progressbar.ProgressBar(max_value=len(store_tour)-2,
                              prefix = "Animation rendering: ")


def animate(i):
    global store_tour, x, y
    ax.clear()
    ax.scatter(x, y)
    ax.plot(x[store_tour[i]], y[store_tour[i]], alpha = 0.5)
    # ax.set_xlabel(f"Temp = {round(tempsteps[i], 2)}")
    bar.update(i+2)

# animate the grid
ani = FuncAnimation(fig, animate, frames=len(store_tour)-1, interval=100)
ani.save("traveling_salesman.mp4")
plt.clf()

