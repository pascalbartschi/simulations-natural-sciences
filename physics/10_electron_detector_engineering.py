import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from time import perf_counter_ns as clock

## definition of global variables
# Electron charge, mass and time units
e = -1.6e-19  # Coulomb
me = 9.11e-31  # kg
h = 10e-12  # s, time step
tEnd = 6e-9  # s
# Board dimensions
N = 100  # number of point in 2D grid
L = 0.01  # cm, side length of box
delta = L / (N - 1)


def VoltageMap(plates):  # defaults after necessary
    # put in last weeks exercise in here
    # number of grid pts

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

    # horizontal
    # fraction_x = 0.9
    # fraction_y = 0.4
    # # set middle array for plate from 0.25 to 0.75
    # U[int(N * fraction_x), int(N * fraction_y): int(N * (fraction_y + 0.5))] = 1000
    # R[int(N * fraction_x), int(N * fraction_y): int(N * (fraction_y + 0.5))] = 0

    for key in plates.keys():
        U, R = placePlate(plates[key]["p1"], plates[key]["p2"], R, U, plates[key]["potential"])

    # set up multiplication frame with same shape as U
    C = np.ones_like(U)
    # output matrix
    M = np.ones_like(U)

    # error application board as seen in exercise session
    B = np.ones_like(U, dtype=bool)
    B[::2, ::2] = False
    B[1::2, 1::2] = False

    while np.max(np.abs(M)) >= 0.1:
        # two step convolution
        ndimage.convolve(U, W, output=C, mode="constant", cval=0)
        np.multiply(R, C, out=M)
        U[B] = U[B] + M[B]

        ndimage.convolve(U, W, output=C, mode="constant", cval=0)
        np.multiply(R, C, out=M)
        U[~B] = U[~B] + M[~B]

    return U


def placePlate(p1, p2, R, U, potential):
    x1, y1 = p1
    x2, y2 = p2
    if x2 < x1:
        t = x2
        x2 = x1
        x1 = t
        t = y2
        y2 = y1
        y1 = t

    a = (y2 - y1) / (x2 - x1)
    j1 = int(x1 / delta)
    j2 = int(x2 / delta)
    l1 = int(y1 / delta)
    l2 = int(y2 / delta)

    n = max(j2 - j1 + 1, l2 - l1 + 1)

    for i in range(n + 1):
        x = x1 + i * (x2 - x1) / n
        y = y1 + a * (x - x1)
        j = int(x / delta)
        l = int(y / delta)
        R[l, j] = 0
        U[l, j] = potential

    return U, R


def generateElectrons(n):
    y = np.linspace(0.6 * L, 0.9 * L, n)  # todo BOX ?
    x = np.zeros_like(y)
    # Take a random angle phi in the range -pi/2 to pi/2
    phi = np.random.uniform(low=-(np.pi / 2), high=(np.pi / 2), size=x.shape)  # either uniform or normal distributed
    # phi = 0
    vx = 1e6 * np.cos(phi)
    vy = 1e6 * np.sin(phi)
    if n == 1:
        vx, vy = np.array([vx]), np.array([vy])
    # vx = 1e6 * np.ones_like(x)
    # vy = np.zeros_like(y)
    return (x, y, vx, vy)


def solverfunc(generateElectrons, numberofElectrons, U):  # todo which order in while loop makes sense?
    x, y, vx, vy = generateElectrons(numberofElectrons)

    marker = np.ones_like(x, dtype="bool")

    electron_count = 0

    pos_storage, det_storage, time_storage = [], [], []

    start = np.ones_like(x) * clock()

    while marker.any():

        x[marker], y[marker], vx[marker], vy[marker] = LeapFrog(accel, x[marker], vx[marker], y[marker], vy[marker], U)

        det_marker = detect_marker(x, y, L)  # todo will detetct same elecvtron multiple times

        fused_marker = np.logical_and(marker, det_marker)

        electron_count += len(x[fused_marker])

        marker = boundary_marker(x, y, L)

        stop = np.ones_like(x) * clock()  # todo how to get only times of detected ones ?

        time = stop[fused_marker] - start[fused_marker]

        if fused_marker.any():
            time_storage.append(list(time))
            # print("m", marker)
            # print("d", det_marker)

            det_storage.append(list(y.copy()[fused_marker]))  # todo check wether correctly marked

        pos_storage.append([x.copy(), y.copy()])  # todo not nicely stored

    storage = {"Position": np.array(pos_storage), "Detection": np.array(sum(det_storage, [])),
               "Time": np.array(sum(time_storage, []))}

    return storage, electron_count


def boundary_marker(x, y, L):
    marker = np.logical_and(np.logical_and(x <= L, x >= 0), np.logical_and(y <= L, y >= 0))

    return marker


def detect_marker(x, y, L):
    detection = np.logical_and((x >= L), np.logical_and(y > 0.1 * L, y < 0.4 * L))

    return detection


# directly taken from assignment 7: orbits
def LeapFrog(accel, x, vx, y, vy, U):
    x_half = x + (0.5 * h * vx)
    y_half = y + (0.5 * h * vy)
    vx_1 = vx + h * accel(U, x, y)[0]
    vy_1 = vy + h * accel(U, x, y)[1]
    x_1 = x_half + (0.5 * h * vx_1)
    y_1 = y_half + (0.5 * h * vy_1)

    return x_1, y_1, vx_1, vy_1


def coordToIndex(x, y):
    j = np.array(x / delta, dtype='int')  # todo DELTA X? same as delta in grid above?
    l = np.array(y / delta, dtype='int')
    return (j, l)


# todo array shapes do not fit together yet

def accel(U, x, y):  # todo
    j, l = coordToIndex(x, y)

    # Make sure j,l are inside grid
    L, J = U.shape
    j = np.maximum(j, np.zeros_like(j))  # todo why element wise??
    j = np.minimum(j, (J - 2) * np.ones_like(j))
    l = np.maximum(l, np.zeros_like(l))
    l = np.minimum(l, (L - 2) * np.ones_like(l))

    t = (x - j * delta) / delta
    u = (y - l * delta) / delta
    U1 = U[l, j]
    U2 = U[l, j + 1]
    U3 = U[l + 1, j]
    U4 = U[l + 1, j + 1]
    ax = -(e / me) / delta * ((1 - u) * (U2 - U1) + u * (U4 - U3))
    ay = -(e / me) / delta * ((1 - t) * (U3 - U1) + t * (U4 - U2))

    return ax, ay


def plot_design(storage):
    U, pos, det, time = storage["U"], storage["Position"], storage["Detection"], storage["Time"]  # extract

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))

    # plot the voltage map
    axs[0, 0].imshow(U, origin="lower")
    CS = axs[0, 0].contour(U, 10, colors="black", linestyles="dotted")
    axs[0, 0].clabel(CS, CS.levels, inline=True, fontsize=13, fmt="%dV", colors="black")

    # plot the electron movements
    for i in range(n):
        axs[0, 1].plot(pos[:, 0, i], pos[:, 1, i])

    axs[0, 1].vlines(x=[0, 0.01, 0.01], ymin=[0, 0, 0.004], ymax=[0.01, 0.001, 0.01], color="red")
    axs[0, 1].vlines(x=0.01, ymin=0.001, ymax=0.004, color="green")
    axs[0, 1].hlines(y=[0, 0.01], xmin=[0, 0], xmax=[0.01, 0.01], color="red")

    axs[1, 0].hist(time, bins=20, density=True)
    axs[1, 1].hist(det, bins=20, density=True)

    # set titles
    axs[0, 0].set_title("Mapped electrostatic potential")
    axs[0, 1].set_title("Electron trajectories")
    axs[0, 1].set_xlabel("x")
    axs[0, 1].set_ylabel("y")
    axs[1, 0].set_title("Distribution of taken time for detector hit")
    axs[1, 0].set_xlabel("Travel time [ns]")
    axs[1, 1].set_xlim(0.001, 0.004)
    axs[1, 1].set_title("Distribution of detection hit positions")
    axs[1, 1].set_xlabel("Position on y-axis [m]")

    # for i in range(2):
    #     for j in range(2):
    #         axs[i, j].set_adjustable("datalim")
    #         axs[i, j].set_aspect("equal")

    # axs.set_ylim(0, 0.01)
    plt.tight_layout()
    plt.savefig("Design.png")
    # plt.show()
    plt.clf()


plates = {'plate1': {'p1': (0.0098, 0.0037), 'p2': (0.0099, 0.0041), 'potential': 6500}}

U = VoltageMap(plates)
n = 1000
storage, count = solverfunc(generateElectrons=generateElectrons,
                            numberofElectrons=n,
                            U=U)
storage["U"] = U

plot_design(storage)

print("Accuracy:", count / n)
