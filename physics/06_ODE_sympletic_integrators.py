import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl


def odeHO(p, q):
    """Harmonic oscillator"""
    dpdt = -q
    dqdt = p

    return np.array([dpdt, dqdt])


def odePendulum(p, q):
    epsilon = 1
    dpdt = - epsilon * np.sin(q)
    dqdt = p

    return np.array([dpdt, dqdt])


def leapFrog(p0, q0, h, odeSystem):
    q12 = q0 + 0.5 * h * p0  # first drift
    dp, dq = odeSystem(p0, q12)
    p1 = p0 + h * dp  # kick
    q1 = q12 + 0.5 * h * p1  # second drift
    return (p1, q1)


def euler(p, q, h, odeSystem):
    dp, dq = odeSystem(p, q)
    p += h * dp
    q += h * dq
    return (p, q)


def rk2MidPoint(p, q, h, odeSystem):
    pandq = np.array([p, q])
    midp, midq = pandq + ((h / 2) * odeSystem(p, q))
    p, q = pandq + (h * odeSystem(midp, midq))
    return (p, q)


def rk4(p, q, h, odeSystem):
    return


# ode Solver
def odeSolver(p0, q0, odeSys, h, nSteps, solverStepFunc):
    """This is a general ODE solver that takes the
    derivative df/dt (dfFunc) and the algorithm for one time
    step (solverStepFunc) as function arguments.
	"""
    pn = p0
    qn = q0
    plist = [p0]
    qlist = [q0]
    for n in range(nSteps):
        pn1, qn1 = solverStepFunc(pn, qn, h, odeSys)
        plist.append(pn1)
        qlist.append(qn1)
        pn = pn1
        qn = qn1
    return (np.array(plist), np.array(qlist))


# # optional for function animation
# def update(frame):
#     global p, q
#     # p, q = leapFrog(p, q, h, odeHO)
#     p, q = euler(p, q, h, odeHO)
#     xdata.append(q)
#     ydata.append(p)
#     ln.set_data(xdata, ydata)
#     return ln,

def plot_phases_HO(subplots, p0, q0, H, h, nSteps):
    ode = odeHO
    solver_list = [euler, rk2MidPoint, leapFrog]

    fig, axs = subplots

    results = {}

    for j, solver in enumerate(solver_list):
        p_array, q_array = odeSolver(p0, q0, ode, h, nSteps, solver)
        results[ode.__name__ + " + " + solver.__name__] = np.array([p_array, q_array])
        axs[0, j].plot(q_array, p_array, color=cmap_harmonic(H))

    # axs[0, 2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return results


def plot_phases_pendulum(subplots, p0, q0, H, h, nSteps):
    ode = odePendulum
    solver_list = [euler, rk2MidPoint, leapFrog]

    fig, axs = subplots

    results = {}

    for j, solver in enumerate(solver_list):
        p_array, q_array = odeSolver(p0, q0, ode, h, nSteps, solver)
        results[ode.__name__ + " + " + solver.__name__] = np.array([p_array, q_array])
        axs[1, j].plot(q_array, p_array, color=cmap_pendulum(H))
        axs[1, j].set_xlim(-10, 10)

    # axs[1, 2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return results


def subplots():
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize = (20, 12))

    # make room for row titles
    right = 0.94  # keeps fraction white vs plots
    top = 0.87
    space = 0.1
    fig.subplots_adjust(right=right, left=right - 0.775, top=top, bottom=top - 0.8,
                        wspace=space, hspace=space + 0.15)

    ode_list = ["Harmonic O.", "Pendulum"]
    solver_list = [euler.__name__, rk2MidPoint.__name__, leapFrog.__name__]

    # annotate titles manually and set axis labels
    for ax, col in zip(axs[0], solver_list):
        ax.annotate(text=col, xy=(0.5, 1.05), annotation_clip=False, fontsize=20, xycoords='axes fraction',
                    ha="center", va="center")

    for ax, row in zip(axs[:, 0], ode_list):
        ax.annotate(text=row, xy=(-0.5, 0.5), annotation_clip=False, fontsize=20, xycoords='axes fraction',
                    ha="left", va="center")
        
    plt.annotate(text="HO & Pendulum behaviour at different initial energies",
                 xy=(0.5, 0.97), annotation_clip=False, fontsize=30,
                 xycoords="figure fraction", ha="center", va="center")

    return (fig, axs)


if __name__ == "__main__":
    h = 0.1
    nSteps = 600

    # colormap = cm.get_cmap(name="rainbow")

    subplots = subplots()

    harmonic_length = 3
    harmonic_p0 = np.zeros(harmonic_length)
    harmonic_q0 = np.linspace(1, 3, harmonic_length)
    harmonic_energy = np.round((1 / 2 * harmonic_p0 ** 2 - np.cos(harmonic_q0)), 2)
    harmonic_experiments = [[i, j, k] for i, j, k in
                            zip(harmonic_p0, harmonic_q0, harmonic_energy)]  # np.array([[0, i] for i in range(3)])
    
    cmap_harmonic = cm.get_cmap("jet", len(harmonic_energy))

    for exp in harmonic_experiments:
        p0, q0, H = exp
        plot_phases_HO(subplots, p0, q0, H, h, nSteps)
    #
    # # plt.colorbar()
    #
    pendulum_length = 50
    pendulum_p0 = np.linspace(-2, 2, pendulum_length)
    pendulum_q0 = np.linspace(-np.pi, np.pi, pendulum_length)
    pendulum_energy = 1 / 2 * pendulum_p0 ** 2 - np.cos(pendulum_q0)
    pendulum_experiments = [[i, j, k] for i, j, k in
                            zip(pendulum_p0, pendulum_q0, pendulum_energy)]  # np.array([[0, i] for i in range(3)])

    cmap_pendulum = cm.get_cmap("jet", len(pendulum_energy))
    

    for exp in pendulum_experiments:
        p0, q0, H = exp
        plot_phases_pendulum(subplots, p0, q0, H, h, nSteps)
        
    fig, axs = subplots
        
    fig.savefig(fname = "pendulum_HO_in_solver_comparison.pdf")

    # colormap = cm.get_cmap(name="rainbow")

    # q0 = 1
    # h = 0.1
    # nSteps = 600
    # result = plot_phases(subplots, p0, q0, h, nSteps)
    # adantage of symplectic is the fixed orbid ( e conserved), orbid/ angle is inreasing in rk2 and euler,
    # pendulum speeds up

# fig1, axs = plt.subplots(2,2)
#
# #initial conditions
# HO: p=0, q=1, q=2, q=3 3 harm
# pendulum p range 2, q range 2 pi
# colorcode by h
# jzs plot everthing on same figure, grid
# Pendulum: q is angle, range is [-pi,pi]
# For circulation you need an additional momentum p /= 0
#
# axs[0,0]: phase plot HO using leap frog
# axs[0,1]: phase plot HO using euler or rk2 or rk4
# axs[1,0]: phase plot pendulum using leap frog
# axs[1,1]: phase plot pendulum using euler or rk2 or rk4
