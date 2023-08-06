

from numba import jit
import numpy as np
#from scipy import ndimage, signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import progressbar

N=150
J=1
N_PER_TEMP = 40 * N * N
TEMP_START = 4
TEMP_END = 0.1
TEMP_FACTOR = 0.98
kb = 1 #1.380649*(10**(-23))

@jit(nopython=True)
def init_grid():
    global N, J, N_PER_TEMP, TEMP_START, TEMP_END, TEMP_FACTOR, kb, N_PER_TEMP

    S = np.random.choice(np.array([-1, 1]), size = (N, N), replace=True) # important to set replace = True, otherwise same is picked for (N,N)

    # E = np.zeros((N,N))

    # making borders/boundary cond.
    S[0, :] = 1
    S[N - 1, :] = 1
    S[:, 0] = 1
    S[:, N - 1] = 1

    T = TEMP_START
    temps = [T]
    while T >= TEMP_END:
        T *= TEMP_FACTOR
        temps.append(T)

    # temps = np.repeat(a = temps, repeats = N_PER_TEMP) # repeat every temp for N_PER_TEMPS times

    return S, np.array(temps)

@jit(nopython=True)
def spin_step(S, beta):
    # global N, J, N_PER_TEMP, TEMP_START, TEMP_END, TEMP_FACTOR, kb

    i,j = np.random.choice(np.arange(0,N-1), size = 2, replace=True)
    dE=2*J*S[i][j]*(S[i-1][j] + S[i+1][j] + S[i][j-1] + S[i][j+1])
    if dE<0:
        S[i][j] *=-1

    else:
        r= np.random.uniform(0,1)
        if r<np.exp(-beta*dE):
            S[i][j] *= -1


    return S

# @jit(nopython=True)
# def sim2():
#     global N_PER_TEMP
#     S, E, tempsteps = init_grid() # add repeat line
#     tempstep_betas = 1 / (kb * tempsteps)
#     # store_S = np.zeros((N, N, len(tempstep_betas)))
#     store_S = []
#     c = -1
#
#     for i in range(len(tempstep_betas)):
#         c += 1
#         S = spin_step(S, tempstep_betas[i])
#         # store_S[:, :, i] = S.copy()
#         # only store
#         # if i % N_PER_TEMP == 0:
#         store_S.append(list(S.copy()))
#         # stop if equilibrium
#         if (S.copy().ravel() == 1).all() or (S.copy().ravel() == -1).all():
#             break
#
#     return store_S, c, i


@jit(nopython=True)
def sim_temps():
    """
    Simulates the Ising model over Temperature range tempsteps
    :return: stored Simulation and temperature steps
    """
    global N_PER_TEMP
    S, tempsteps = init_grid()
    store_S = [S.copy()]
    magnetization = []

    for TEMP in tempsteps[1:]:
        # safety stop when simulation found equilibrium
        if (S.ravel() == 1).all() or (S.ravel() == -1).all():
            break
        beta = 1 / (kb * TEMP)
        # we could also just jit this loop, bc this is time intensive one
        for i in range(N_PER_TEMP):
            S = spin_step(S, beta)
        store_S.append(S.copy())
        magnetization.append(S.sum()/N**2)




    return store_S, tempsteps, magnetization

start = time.time()
#store_S, steps , i_eq = sim()
store_S, tempsteps, tot_magnetization = sim_temps()
store_S.append(store_S[-1].copy()) # due to plotting last frame issues in animate
store_S = np.array(store_S)
stop = time.time()
print(f"Compiling simulation took {round(stop - start, 2)} seconds.")
fig, ax = plt.subplots(1, 1)
bar = progressbar.ProgressBar(max_value=len(store_S)-1,
                              prefix = "Animation rendering: ")


def animate(i):
    global store_S
    ax.clear()
    ax.imshow(store_S[i, :, :],
              cmap="binary",
              origin='lower',
              vmin = -1,
              vmax = 1)
    ax.set_xlabel(f"Temp = {round(tempsteps[i], 2)}")
    bar.update(i+1)

# animate the grid
ani = FuncAnimation(fig, animate, frames=len(store_S)-1, interval=100)
ani.save("spin.mp4")
plt.clf()

# plot the mean magnetization over temperature
plt.figure()
plt.xlabel("T")
plt.ylabel("mean magnetization")
plt.scatter(tempsteps[:len(tot_magnetization)][::-1], tot_magnetization)
plt.savefig("magnetization.png")













