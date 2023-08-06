import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

"""
This script serves simulating circadiane rythms with a simply parameterized model system of ODEs
"""

def simulation_framework(init, model, parms, method, timesteps, h):
    """
    :param init: tuple of initial system condition
    :param model: parameterized equation system model
    :param parms: parameter values input for model
    :param method: method used to deterministacally solve simulation values
    :param timesteps: number of steps to simulate
    :return: array of states in time
    """

    R1, P1, P1i, R2, P2, P2i = init

    statedict = {"time": np.linspace(0, h, timesteps),
                 "R1": [R1],
                 "P1": [P1],
                 "P1i": [P1i],
                 "R2": [R2],
                 "P2": [P2],
                 "P2i": [P2i]}

    state = {"R1": R1,
             "P1": P1,
             "P1i": P1i,
             "R2": R2,
             "P2": P2,
             "P2i": P2i}

    for t in statedict["time"][:-1]:

        state = method(state = state, model = model, parms = parms, h = h)

        for key in list(statedict.keys())[1:]:
            statedict[key].append(state[key])

    # np.append(statedict["time"], statedict["time"][-1] + h)

    return statedict




def model(state, parms):
    s_R1P1i, d_R1, s_R1P2i, s_P1R1, d_P1, s_P1iP1, d_P1i, s_P1iP2i, s_R2P2i, d_R2, s_R2P1i, s_P2R2, d_P2, s_P2iP2, d_P2i, s_P2iP1i, n_Hill = parms

    dR1_dt  = s_R1P1i * (1 - Hill(x = state["P1i"], n = n_Hill)) - d_R1 * state["R1"]
    dP1_dt  = s_P1R1 * state["R1"] - d_P1 * state["P1"]
    dP1i_dt = s_P1iP1 * state["P1"] - d_P1i * state["P1i"] + s_P1iP2i * Hill(x = state["P2i"], n = n_Hill)
    dR2_dt  = -d_R2 * state["R2"] + s_R2P1i * (1 - Hill(x = state["P1i"], n = n_Hill))
    dP2_dt  = s_P2R2 * state["R2"] - d_P2 * state["P2"]
    dP2i_dt = s_P2iP2 * state["P2"] - d_P2i * state["P2i"]

    return np.array([dR1_dt, dP1_dt, dP1i_dt, dR2_dt, dP2_dt, dP2i_dt])

def Hill(x, n, K = 1):
    """Hill equation"""
    return x ** n / (K + x ** n)

def euler(state, model, parms, h):

     deltas_x_h = model(state, parms) * h

     for i, key in enumerate(state.keys()):
         state[key] += deltas_x_h[i]

     return state

def mid_RK2(state, model, parms, h):

    deltas_x_h_mid = model(state, parms) * (h/2)

    statemid = state.copy() # mid states is just temporary for mid calculation

    for i, key in enumerate(state.keys()):
        statemid[key] += deltas_x_h_mid[i]

    deltas_x_h = model(statemid, parms) * h

    for i, key in enumerate(state.keys()):
        state[key] += deltas_x_h[i]

    return state


def append2state(state):
    pass


# definition of parameter values

s_R1P1i = 0.5
d_R1 = 0.2
s_R1P2i = np.nan
s_P1R1 = 0.5
d_P1 = 0.2
s_P1iP1 = 0.25
d_P1i = 0.4
s_P1iP2i = 0.25
s_R2P2i = np.nan
d_R2 = 0.4
s_R2P1i = 0.25
s_P2R2 = 0.5
d_P2 = 0.2
s_P2iP2 = 0.5
d_P2i = 0.2
s_P2iP1i = np.nan
n_Hill = 9

parms = (s_R1P1i,
         d_R1,
         s_R1P2i,
         s_P1R1,
         d_P1,
         s_P1iP1,
         d_P1i,
         s_P1iP2i,
         s_R2P2i,
         d_R2,
         s_R2P1i,
         s_P2R2,
         d_P2,
         s_P2iP2,
         d_P2i,
         s_P2iP1i,
         n_Hill)

# definition of initial state

R1 = 0.1
P1 = 1
P1i = 0
R2 = 0.1
P2 = 1
P2i = 0

init = (R1,
        P1,
        P1i,
        R2,
        P2,
        P2i)

euler_sim = simulation_framework(init = init,
                                 model = model,
                                 parms = parms,
                                 method = euler,
                                 timesteps = 1000,
                                 h = 0.1)

RK2_sim = simulation_framework(init = init,
                                 model = model,
                                 parms = parms,
                                 method = mid_RK2,
                                 timesteps = 1000,
                                 h = 0.1)


def plot_sim(name, sim):
    plt.figure(figsize = (7, 7))

    for key in list(sim.keys())[1:]:
        plt.plot(sim["time"], sim[key], label = key)

    plt.legend()
    plt.title(f"Concentrations over time of {name}")
    plt.xlabel("time")
    plt.ylabel("c")
    plt.savefig(name + ".pdf")
    # plt.show()

# measure robustness of system: idea, compare mean and variance of data for oscillations

def robustness(sim):
    """idea find amplitude of oscillations and its constance"""
    # a_var = np.array([])
    # random = np.random.uniform(low = 0.1, high = 0.7, size = len(parms)) # new parameterset

    min_i = sig.argrelmin(np.array(sim["P1"]))
    max_i = sig.argrelmax(np.array(sim["P1"]))
    maxs = np.array(sim["P1"])[max_i]
    mins = np.array(sim["P1"])[min_i]

    if len(mins) > len(maxs):
        mins = mins[:len(maxs)]
    elif len(mins) < len(maxs):
        maxs = maxs[:len(mins)]

    amplitudes = maxs - mins

    return amplitudes






    for key in sim.keys():
        local_var = sim[key].var
        local_mean = sim[key].mean


def dict2array(sim):

    rows = len(sim.keys())
    cols = len(sim["time"])
    a = np.array(sim).reshape((rows, cols))

    return a


# if __name__ == "__main__":
#     for name, sim in zip(["Euler", "RK2"], [euler_sim, RK2_sim]):
#         plot_sim(name, sim)
#
#     print("Execution finished")

