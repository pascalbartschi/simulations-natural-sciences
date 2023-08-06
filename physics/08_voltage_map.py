import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# number of grid pts
N = 100

# def omega
w = 2 / (1 + (np.pi / N))

# zero grid: Voltage is  going to be put in here
U = np.zeros(shape=(N, N))

# weights for convolution --> correction
W = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

# boundary conditions / box borders
R = np.ones_like(U) * w / 4
R[0, :] = 0
R[-1, :] = 0
R[:, 0] = 0
R[:, -1] = 0

# set middle array for plate from 0.25 to 0.75
U[int(N * 0.5), int(N * 0.25): int(N * 0.75)] = 1000
R[int(N * 0.5), int(N * 0.25): int(N * 0.75)] = 0

# set up multiplication frame with same shape as U
C = np.ones_like(U)
# output matrix
M = np.ones_like(U)

# error application board as seen in exercise session
B = np.ones_like(U, dtype=bool)
B[::2, ::2] = False
B[1::2, 1::2] = False

# count index
count = 0
# iterate until voltage difference < 0.1

border = 0.1

while np.max(np.abs(M)) >= border:
    # two step convolution
    ndimage.convolve(U, W, output=C, mode="constant", cval=0)
    np.multiply(R, C, out=M)
    U[B] = U[B] + M[B]

    ndimage.convolve(U, W, output=C, mode="constant", cval=0)
    np.multiply(R, C, out=M)
    U[~B] = U[~B] + M[~B]

    count += 1

# plot the convolved voltage grid
fig, ax = plt.subplots(figsize=(10, 10))

ax.imshow(U, cmap="jet", origin="lower")
CS = ax.contour(U, 10, colors="black", linestyles="dotted")
ax.clabel(CS, CS.levels, inline=True, fontsize=13, fmt="%dV", colors="black")
ax.set_title("Voltage cmap with w = {} and {} iterations".format(round(w, 2), count))
plt.savefig("Voltage_Map.png")
plt.show()
