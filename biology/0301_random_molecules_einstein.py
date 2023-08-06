import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time as tim


D = 0.1
time = 10
dt = 0.01
no_molecules = 100


# def Px(time, t0, x, x0, D):
#     return 1 / (4 * np.pi * D * (time - t0)) ** 0.5 * np.exp(-1 * (x - x0) ** 2 * 4 * )

def sim(D, time, dt, no_molecules):
    """
    Simulation to compute molecule movement accroding to normal distribution
    :param D: Diffusion constant
    :param time: total time o
    :param dt: delta t
    :param no_molecules: number of molecules in array
    :return: movements
    """
    # initialize array
    array = np.zeros((no_molecules, int(time / dt) + 1))
    std = (2 * D * dt) ** 0.5  # t - t0 is always dt

    # in each t a new value with mu from t-1 is drawn from Nx
    for t in range(int(time/dt)):
        mu = array[:,t]
        array[:, t+1] = np.random.normal( loc = mu, scale = std,size = no_molecules)

    return array, np.absolute(np.diff(a = array, axis = 1)).sum(axis = 1) # np.absolute(array).sum(axis=1)
# plt.show()


dts = np.array([10 ** i for i in range(-5, 2)])
mean_dists = np.zeros(len(dts))
arrays = {} # np.zeros((no_molecules, int(time / dt) + 1 , len(dts)))

for i, dt in enumerate(dts):
    a, dist = sim(D  = D, time  = time, dt = dt, no_molecules  = no_molecules)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (20, 10))
    sns.histplot(a[:, -1], ax = ax1)
    ax1.set_title(f"Distribution of path lengths for dt = {dt}")
    sns.histplot(dist, ax = ax2)
    ax2.set_title(f"Distrubution of end states for dt = {dt}")
    filename = f"Histograms_dt-{dt}.png"
    plt.savefig(filename)
    mean_dists[i] = dist.sum()
    arrays[str(dt)] = a

# bins with delta x


# fig, axs = plt.subplots(5, 1, figsize = (20, 4))
# for i in len(axs):
#     sns.histplot()

# plt.plot(dts, mean_dists)
# plt.xlabel("dt")
# plt.ylabel("mean")
# plt.savefig("mean_dist.png")






# plt.hist(array[:, -1], density = True)
# sns.histplot(array[:, 1], kde = True)
# plt.hist(sum_dist)
# plt.plot()
