import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import astropy.time
import dateutil.parser


# method to calculate the julian time


#
# dt = dateutil.parser.parse('31.10.2022')
# time = astropy.time.Time(dt)
# time.jd


# read file with np or pandas


def readPlanets(filename, N=-1):
    # Loading text files with Pandas
    df = pd.read_csv(filename, sep=',', header=None,
                     names=['name', 'm', 'x', 'y', 'z', 'vx', 'vy', 'vz'])

    # Data is now in a Pandas dataframe
    # print(df)
    # ways to assign column to arrays:

    name = np.array(df.loc[:, 'name'])
    m = np.array(df.loc[:, 'm'])
    r = np.array(df.loc[:, 'x':'z'])  # index of planet returns the position of planet
    v = np.array(df.loc[:, 'vx':'vz'])

    # not really necessary just for debugging

    if N > 0:
        name = name[0:N - 1]
        m = m[0:N - 1]
        r = r[0:N - 1]
        v = v[0:N - 1]

    return (name, r, v, m)


def center_of_mass(m, r):  # todo calculate k**2 yourself

    mass_center = 0
    for ri, mi in zip(r, m):
        mass_center += ri * mi
    mass_center = mass_center / sum(m)

    return mass_center


def solver_func(m0, r0, v0, h, start_date, end_date):  ## todo center of mass movement when rest works

    ds = dateutil.parser.parse(start_date)
    time = astropy.time.Time(ds)
    start = time.jd
    de = dateutil.parser.parse(end_date)
    time = astropy.time.Time(de)
    end = time.jd

    NSteps = int((end - start) / h)

    r_sim = [r0]
    m, r, v = m0, r0, v0
    c_sim = [center_of_mass(m0, r0)]  # center of mass array update

    for i in range(0, NSteps):  # todo julian start end how ?

        rn, vn = LeapFrog(accel, m, r, v, h)
        r, v = rn, vn  ## velocity and position update

        c_sim.append(center_of_mass(m, r))

        r_sim.append(rn)

    return np.array(r_sim), np.array(c_sim)


def LeapFrog(acceleration, m, r, v, h):  # leapfrog with negative h from today to 1960
    r_half = r + (0.5 * h * v)
    v_1 = v + h * acceleration(m, r_half, v)  # typo script stadel
    r_1 = r_half + (0.5 * h * v_1)

    return r_1, v_1


def accel(m, r, v):
    k_sq = 0.01720209895 ** 2  ## k squared # defines time step days, h = is 4 days, 365 * h = 1 would be earth orbit
    a = np.zeros(shape=np.shape(v))
    N = len(v)
    for i in range(0, N):
        for j in range(i + 1, N):
            norm = np.linalg.norm(r[j] - r[i])  # norming the unit vector
            F = ((k_sq * m[i] * m[j]) / (norm ** 3)) * (
                    r[j] - r[i])  ## last term: unit vector, absolute in sense of vector length
            a[i] += F / m[i]
            a[j] -= F / m[j]

    return a


name, r0, v0, m0 = readPlanets("SolSystData.dat")
h = -1  # iterate backwards
start = "27.10.2022"  # my birthday
end = "27.10.1960"
simulation, centers = solver_func(m0, r0, v0, h, start, end)

# plot data
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(20, 30))

# make room for row titles
right = 0.95  # keeps fraction white vs plots
top = 0.87
space = 0.25
fig.subplots_adjust(right=right, left=right - 0.775, top=top, bottom=top - 0.8,
                    wspace=space, hspace=0)

# column titles
cols = ["Planet orbits", "Planet orbits", "Center of Mass vs. Sun"]
for ax, col in zip(axs[0], cols):
    ax.annotate(text=col, xy=(0.5, 1.05), annotation_clip=False, fontsize=15, xycoords='axes fraction',
                ha="center", va="center")

# set row titles
rows = ["y vs. x", "z vs. x"]
for ax, row in zip(axs[:, 0], rows):
    ax.annotate(text=row, xy=(-0.6, 0.5), annotation_clip=False, fontsize=15, xycoords='axes fraction',
                ha="left", va="center")

title = "Planet movement from {} back to {}".format(start, end)
plt.annotate(text=title, xy=(0.5, 0.97), annotation_clip=False, fontsize=20, xycoords="figure fraction",
             ha="center", va="center")

# ax.set_aspect("equal")

# plot first five planets
for i in range(0, 5):
    axs[0, 0].plot(simulation[:, i, 0], simulation[:, i, 1], label=name[i])  # plot x vs. y
    axs[1, 0].plot(simulation[:, i, 0], simulation[:, i, 2])  # plot x vs. z

# plot least four planets
for i in range(5, 9):
    axs[0, 1].plot(simulation[:, i, 0], simulation[:, i, 1], label=name[i])  # plot x vs. y
    axs[1, 1].plot(simulation[:, i, 0], simulation[:, i, 2])  # plot x vs. z

# plot center vs. sun
relative_center = centers - simulation[:, 0, :]
axs[0, 2].plot(relative_center[:, 0], relative_center[:, 1], label="M-center")
axs[1, 2].plot(relative_center[:, 0], relative_center[:, 2])
axs[0, 2].plot(simulation[:, 0, 0], simulation[:, 0, 1], label="Sun")
axs[1, 2].plot(simulation[:, 0, 0], simulation[:, 0, 2])

axs[0, 0].legend(loc="lower left")
axs[0, 1].legend(loc="lower left")
axs[0, 2].legend(loc="lower left")

for ax in [axs[0, 0], axs[0, 1], axs[0, 2]]:
    # ax.set_aspect("equal")
    ax.set_xticks([])

# plt.savefig(title.replace(" ", "_") + ".png")

# center of mass plot: center = centers - simualtion[:,0,:] center of mass relative to sun

# axs.set_aspect()
