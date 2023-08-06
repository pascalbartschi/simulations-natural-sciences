import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter_ns as clock


# without gillespie
def run_nogill(init, L, dx, tot_t, dt, kd, ks, k1, D):

    # init conditions
    A = np.zeros(int(L / dx))
    A[len(A) // 2] = init
    # param


    for t in range(int(tot_t / dt)):
        # diffusion
        r = np.random.uniform(0, 1, size=1)[0]
        # r = 0.0006
        f1 = np.cumsum(A[:-1] * d * dt)
        f2 = np.cumsum(A[1:] * d * dt)
        diff_right = np.where(r < f1)[0]
        if any(diff_right):
            i_right = diff_right[0]
            A[i_right + 1] += 1
            A[i_right] -= 1
        else:
            diff_left = np.where(r < (f1[-1] + f2))[0]
            if any(diff_left):
                i_left = diff_left[0] + 1
                A[i_left - 1] += 1
                A[i_left] -= 1
        # diff_left = np.where(np.random.uniform(0, 1, size = A[:-1].shape) <= 0.01) + 1
        # A[:-1][diff_left] += 1
        # diff_right = np.where(np.random.uniform(0, 1, size = A[1:].shape) <= 0.01) + 1
        # A[1::-1][diff_right] += 1

    return A




def f_prosperity(A, dx, kd, ks, k1, k2 = 0, B = None):
    if not B:
        B = np.zeros_like(A)
    # reaction of Ai with Ai
    # y = (A * (kd + B * k1 + A * k1 + (A - 1) * k2)).sum() \
    # # Synthesis of Ai
    # len(A) * ks * (dx / L) \
    # diffusion from left to right, right to left
    y = (A[:-1] * d).sum() + (A[1:] * d).sum()

    return y

def gillespie(A, dx, kd, ks, k1, f = f_prosperity):

    return 1/f(A, dx, kd, ks, k1) * np.log(1/np.random.uniform(0, 1))


def run_gill(init, L, dx, tot_t, dt, kd, ks, k1, D):
    # init conditions
    # init = 100
    A = np.zeros(int(L / dx))
    A[len(A)//2] = init
    steps = int(tot_t / dt)
    # param d
    d = D / dx ** 2
    # time
    t = 0

    while t <= tot_t:
        # calc dt
        dt = gillespie(A, dx, kd, ks, k1)
        # add to time
        t += dt
        # calculate weights
        # p = (np.cumsum(A[:-1]) + np.cumsum(A[-1:0:-1])) * d * dt
        # p = np.cumsum(A[:-1] + A[-1:0:-1]) * d * dt
        pA = A.copy().astype(float)
        pA[0], pA[-1] = 0.5 * np.array([A[0], A[-1]])
        pos = np.random.choice(a = np.arange(0, L/dx), p = pA / pA.sum()).astype(int)
        if pos == len(pA)-1:
            A[-2] += 1
            A[-1] -= 1
        elif pos == 0:
            A[1] += 1
            A[0] -= 1
        else:
            dir = np.random.choice(a=[-1, 1])
            A[pos+dir] += 1
            A[pos] -= 1
        # r = np.random.uniform(0, 1, size=1)[0]
        # dir = np.random.choice([0,1])
        # r = 0.0006
        # if dir == 0:
        # diff_right = np.where(r < np.cumsum(A[:-1] * d * dt))[0]
        # if any(diff_right):
        #     i_right = diff_right[0]
        #     A[i_right + 1] += 1
        #     A[i_right] -= 1
        # # else:
        # diff_left = np.where(r < np.cumsum(A[-1:0:-1] * d * dt))[0]
        # if any(diff_left):
        #     i_left = diff_left[1] + 1
        #     A[i_left - 1] += 1
        #     A[i_left] -= 1

    return A

# simulation constants
dt = 0.01
tot_t = 1000
D = 0.4
dx = 10
L = 1000

# parameters
ks = 0.1
kd = 0.1
k1 = 0.1
k2 = 0.1
d = D / dx ** 2

# init conditions
init = 1000
A = np.zeros(int(L / dx))
A[len(A)//2] = init

# check the dt
print("dt for nogill:", (A[:-1] * d * dt).sum() + (A[1:] * d * dt).sum())

start = clock()
A1 = run_nogill(init, L, dx, tot_t, dt, kd, ks, k1, D)
stop = clock()
time_nogill = (stop - start) / 1e9

start = clock()
A2 = run_gill(init, L, dx, tot_t, dt, kd, ks, k1, D)
stop = clock()
time_gill = (stop - start) / 1e9

fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10, 5))
ax1.plot(A1)
ax1.set_title(f"Simulation P(tot) = 0.01 for {tot_t} time took {round(time_nogill, 2)} sec")
ax2.plot(A2)
ax2.set_title(f"Gillespie Simulation for {tot_t} time took {round(time_gill, 2)} sec")
plt.tight_layout()
plt.savefig(f"Compare_Gillespie_{tot_t}.png")
plt.show()

# s = 1e9ns

# sns.histplot(A1, bins = int(L / dx) + 1, kde = True)
# plt.show()
# plt.plot()