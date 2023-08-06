import numpy as np


def bcd_propensity(A, dx, kd, ks, solve = None):

    degradation = (A * kd).sum()
    # # Synthesis of Ai
    production = ks # len(A[:10]) * ks * (dx / len(A[:10]))
    # diffusion from left to right, right to left
    diffusion = (A[:-1] * d).sum() + (A[1:] * d).sum()


    return degradation + production + diffusion

def gillespie(A, dx, kd, ks, f = bcd_propensity):

    return 1/f(A, dx, kd, ks) * np.log(1/np.random.uniform(0, 1))

# params
L = 450 # in mikrometers #10e-6 meter
dx = L / 100 # 5 mikrometers per nuclei ->
kd = 0.0002 # 1/ s
T_bcd = 40 * 1e-9 * 1e-15 # mol / mikrometer ** 3
V = dx ** 3 # mikrometer**3
# threshold of overall bcd concentration in moles / m**3 * 1e-18 * 10e6 for
# one microM is 10e-6 mol/L -> 10e-2 mol/m**3
# in nM 4e-5 mol / m**3 -> 4e-5 * 10e18 mol / mikrometer ** 3
D = 4

no_molecules = V * T_bcd * 6e23 # maximum number of molecules that one nucelas can contain
# sim frame


prod_L  = int(10 / dx)
tot_t = 10000
t = 0
d = D / dx ** 2
# moles are molecules per liter
# init
A = np.zeros(int(L / dx))
ks = 10
# T = tot_t * (ks - kd)

# ticklist: production works:


while tot_t > t:
    ## update timeline with gillespie
    dt = gillespie(A, dx, kd, ks)
    # print(dt)
    t += dt
    ## draw a number of a uniform
    r = np.random.uniform(0, 1)

    ## cumulative prosperities of respective events:
    fdl = np.cumsum(A[:-1] * d) / bcd_propensity(A, dx, kd, ks)
    fdr = np.cumsum(A[1:] * d) / bcd_propensity(A, dx, kd, ks) + fdl[-1]
    fpr = np.cumsum(np.ones(len(A[:prod_L])) * ks/len(A[:prod_L])) / bcd_propensity(A, dx, kd, ks) + fdr[-1]
    fde = np.cumsum(A * kd) / bcd_propensity(A, dx, kd, ks) + fpr[-1]
    # diffusion right
    dl = np.where(r < (fdl))[0]
    # print("dl", dl)
    # diffusion left
    dr = np.where(r < (fdr))[0]
    # print("dr", dr)
    pr = np.where(r < (fpr))[0]
    # print("pr", pr)
    # degradation
    de = np.where(r < (fde))[0]
    # print("de", de)

    if len(dl) > 0:
        A[dl[0] + 1] += 1
        A[dl[0]] -= 1
    elif len(dr) > 0:
        A[dr[0]] += 1
        A[dr[0] + 1] -= 1
    elif len(pr) > 0:
        A[pr[0]] += 1
    elif len(de) > 0:
        A[de[0]] -= 1
    else:
        raise Exception("No event occured. Please ensure occurence at every timestep!")

import matplotlib.pyplot as plt
plt.figure()
plt.plot(np.arange(len(A)) * dx, A)
plt.xlabel("x")
plt.ylabel("[bcd]")
plt.title(f"Bicoid concentration over larvae at t = {tot_t}")
plt.savefig(f"Bicoid_gradient_t-{tot_t}.png")


