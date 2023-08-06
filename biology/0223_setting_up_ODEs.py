import numpy as np
import matplotlib.pyplot as plt

# EXERCISE 1

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


def mid_RK2(state, model, parms, h):

    deltas_x_h_mid = model(state, parms) * (h/2)
    statemid = state.copy() # mid states is just temporary for mid calculation

    for i, key in enumerate(state.keys()):
        statemid[key] += deltas_x_h_mid[i]

    deltas_x_h = model(statemid, parms) * h

    for i, key in enumerate(state.keys()):
        state[key] += deltas_x_h[i]

    return state

def ODEs1(state, parms):
    """ODE system for EXERCISE 1"""
    dA = -parms["kb"] * state["A"] * state["B"] + parms["k-b"] * state["AB"]
    dB = -parms["kb"] * state["A"] * state["B"] + parms["k-b"] * state["AB"]
    dAB = parms["kb"] * state["A"] * state["B"] - parms["k-b"] * state["AB"]

    return np.array([dA, dB, dAB])
