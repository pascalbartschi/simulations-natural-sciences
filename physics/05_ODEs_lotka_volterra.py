import numpy as np
import matplotlib.pyplot as plt


def odeSolver(t0, y0, dfFunc, h, nSteps, solverStepFunc):
    """This is a general ODE solver that takes the
    derivative df/dt (dfFunc) and the algorithm for one time
    step (solverStepFunc) as function arguments.
	t0 = Initial time
	y0 = Initial function values (array)
	nSteps = Total number of integration steps
	solverStepFunc = Euler Method, Midpoint RK or RK4
    """
    yn = y0
    tn = t0
    tlist = [t0]
    ylist = [y0]
    for n in range(nSteps):
        yn1 = solverStepFunc(tn, yn, h, dfFunc)
        tn1 = tn + h
        tlist.append(tn1)
        ylist.append(yn1)
        tn = tn1
        yn = yn1
    return (np.array(tlist), np.array(ylist))


def eulerStep(tn, yn, h, dfdt):
    dy = h * dfdt(tn, yn)
    yn1 = yn + dy
    return yn1


def MidPointRK2Step(tn, yn, h, dfdt):
    yn0_5 = yn + ((h / 2) * dfdt(tn, yn))
    yn1 = yn + (h * dfdt((tn + (h / 2)), yn0_5))

    return yn1


def RK4Step(tn, yn, h, dfdt):
    k1 = h * dfdt(tn, yn)
    k2 = h * dfdt(tn + h / 2, yn + k1 / 2)
    k3 = h * dfdt(tn + h / 2, yn + k2 / 2)
    k4 = h * dfdt(tn + h, yn + k3)

    yn1 = yn + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6

    return yn1


def LotkaVolterra(t, y):
    """Implements the Lotka Volterra System dm/dt and dfox/dt
    where y=(m,f), dy/dt=(dm/dt, df/dt)"""

    dmdt = (2 * y[0]) - (0.01 * y[0] * y[1])
    dfdt = (-1.06 * y[1]) + (0.02 * y[0] * y[1])

    return np.array([dmdt, dfdt])


def simulation(h, nSteps, y0):
    """simulation framework that uses 3 different Step method to evaluate"""
    t0 = 0

    method_dynamics = {'parameters': {'h': h, 'nSteps': nSteps, 'y0': y0}}

    t, y = odeSolver(t0, y0, LotkaVolterra, h, nSteps, eulerStep)
    method_dynamics[eulerStep.__name__] = {'timeline': t, 'dynamics': y}

    t, y = odeSolver(t0, y0, LotkaVolterra, h, nSteps, MidPointRK2Step)
    method_dynamics[MidPointRK2Step.__name__] = {'timeline': t, 'dynamics': y}

    t, y = odeSolver(t0, y0, LotkaVolterra, h, nSteps, RK4Step)
    method_dynamics[RK4Step.__name__] = {'timeline': t, 'dynamics': y}

    return method_dynamics


def plot_dynamics(experiment, figsave=False):
    """plots a simulation and saves figure when figsave = True"""
    cols = [str(name) for name in experiment.keys()]
    del cols[0]
    rows = ["Dynamics", "Phase plot"]
    fig, axs = plt.subplots(nrows=len(rows),
                            ncols=len(cols),
                            figsize=(20, 12))  # constrained_layout=True # adjust layout throughout plotting

    # make room for row titles
    right = 0.94  # keeps fraction white vs plots
    top = 0.87
    space = 0.1
    fig.subplots_adjust(right=right, left=right - 0.775, top=top, bottom=top - 0.8,
                        wspace=space, hspace=space + 0.15)

    # annotate titles manually and set axis labels
    for ax, col in zip(axs[0], cols):
        ax.annotate(text=col, xy=(0.5, 1.05), annotation_clip=False, fontsize=15, xycoords='axes fraction',
                    ha="center", va="center")

    for ax, row in zip(axs[:, 0], rows):
        ax.annotate(text=row, xy=(-0.6, 0.5), annotation_clip=False, fontsize=15, xycoords='axes fraction',
                    ha="left", va="center")
    title = ("Lotka Volterra h={} nSteps={} y0={}").format(experiment['parameters']['h'],
                                                           experiment['parameters']['nSteps'],
                                                           experiment['parameters']['y0'])
    plt.annotate(text=title, xy=(0.5, 0.97), annotation_clip=False, fontsize=20, xycoords="figure fraction",
                 ha="center", va="center")
    # plot experiments
    for i, method in enumerate(experiment.keys()):
        if str(method) == "parameters":
            continue
        t, y = experiment[str(method)]['timeline'], experiment[str(method)]['dynamics']

        axs[0, i - 1].plot(t, y[:, 0], color="green", label="mice")
        axs[0, i - 1].plot(t, y[:, 1], color="red", label="foxes")
        axs[0, i - 1].set(xlabel="time [steps]", ylabel="abundance [individuals]")
        axs[1, i - 1].plot(y[:, 0], y[:, 1], color="purple", label="phase")
        axs[1, i - 1].set(xlabel="mice [individuals]", ylabel="foxes [individuals]")

    # share axis ticks and labels
    # axs[0, 2].get_shared_x_axes().join(axs[0, 1], axs[0, 2], axs[0, 0])
    # axs[1, 2].get_shared_x_axes().join(axs[1, 1], axs[1, 2], axs[1, 0])
    # axs[0, 0].get_shared_y_axes().join(axs[0, 0], axs[0, 1], axs[0, 2]) # somehow always applies small axis
    # set up lims manually
    for i in range(2):
        # yaxis
        ylb, yub = axs[i, 2].get_ylim()
        axs[i, 0].set_ylim(ylb, yub)
        axs[i, 1].set_ylim(ylb, yub)
        axs[i, 1].yaxis.set_ticklabels([])
        axs[i, 1].set_ylabel('')
        axs[i, 2].yaxis.set_ticklabels([])
        axs[i, 2].set_ylabel('')
        # xaxis
        xlb, xub = axs[i, 2].get_xlim()
        axs[i, 0].set_xlim(xlb, xub)
        axs[i, 1].set_xlim(xlb, xub)

    axs[0, 2].legend(loc="upper right")
    axs[1, 2].legend(loc="upper right")

    if figsave == True:
        fig.savefig(title.replace(" ", "_") + ".pdf")

    # alternative to adjust space between plots:
    # fig.tight_layout()
    return experiment


## set figsave = False if safe unwanted

if __name__ == "__main__":
    experiment1 = plot_dynamics(simulation(h=0.1, nSteps=600, y0=np.array([100, 15])),
                                figsave=True)  # asked from lecture
    experiment2 = plot_dynamics(simulation(h=0.1, nSteps=600, y0=np.array([1000, 15])),
                                figsave=True)  # asked from lecture
    experiment3 = plot_dynamics(simulation(h=0.1, nSteps=600, y0=np.array([1, 15])),
                                figsave=True)  # asked from lecture
    experiment4 = plot_dynamics(simulation(h=0.1, nSteps=600, y0=np.array([100, 100])),
                                figsave=True)  # asked from lecture
    experiment5 = plot_dynamics(simulation(h=0.01, nSteps=6000, y0=np.array([100, 15])),
                                figsave=True)  # asked from lecture
    experiment6 = plot_dynamics(simulation(h=0.1, nSteps=3000, y0=np.array([100, 15])),
                                figsave=True)  # longer simulation

# todo: ask why lims cant be set like that above
