from heapq import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

## globals
# number of nearest neighbours
NN = 32

class particle:

    def __init__(self, r, gamma = 2, P = 1, m = 0.0, v = None, e = None):
        self.r = r  # positionn of the particle [x, y]
        self.m = m  # mass
        self.v = v if v is not None else np.zeros(2)# velocity
        self.vpred = None # np.zeros(2)# predicted velocity
        self.P = P  # pressure
        self.rho = 0.0  # density of the particle
        self.pq = pq(N = NN) # priority queue, NN is hardcoded!
        self.e = None  # energy
        self.epred = None # predicted energy
        self.c = None  # speed of sound
        self.pi_ab = None
        self.dv = np.zeros(2)
        self.de = 0.
        self.gamma = gamma

    def reset_pq(self):

        self.pq = pq(N=NN)

    @staticmethod
    def neighbor_search(pq, root, particles, r, rOffset):
        """Do a nearest neighbor search for particle at  'r' in the tree 'root'
           using the priority queue 'pq'. 'rOffset' is the offset of the root
           node from unit cell, used for periodic boundaries.
           'particles' is the array of all particles.
        """
        if root is None:
            return

        ri = r + rOffset
        # print(ri)
        if root.pLower is not None and root.pUpper is not None:
            d2_lower = root.pLower.celldist2(ri)  # dist2(root.pLower.rc, ri) # dist2 is some other funcion ig
            d2_upper = root.pLower.celldist2(ri)  # dist2(root.pUpper.rc, ri)
            if d2_lower <= d2_upper:
                if root.pLower.celldist2(ri) < pq.key():
                    neighbor_search(pq, root.pLower, particles, r, rOffset)
                if root.pUpper.celldist2(ri) < pq.key():
                    neighbor_search(pq, root.pUpper, particles, r, rOffset)
            else:
                if root.pUpper.celldist2(ri) < pq.key():
                    neighbor_search(pq, root.pUpper, particles, r, rOffset)
                if root.pLower.celldist2(ri) < pq.key():
                    neighbor_search(pq, root.pLower, particles, r, rOffset)
        elif root.pLower is not None:
            neighbor_search(pq, root.pLower, particles, r, rOffset)
        elif root.pUpper is not None:
            neighbor_search(pq, root.pUpper, particles, r, rOffset)
        else:  # root is a leaf cell
            for j in range(root.iLower, root.iUpper + 1):
                d2 = dist2(particles[j].r, ri)
                if d2 < pq.key():
                    pq.replace(d2, particles[j], particles[j].r - rOffset, particles[j].m)

    def neighbor_search_periodic(self, root, particles, period):
        # walk the closest image first (at offset=[0, 0])
        for y in [0.0, -period[1], period[1]]:
            for x in [0.0, -period[0], period[0]]:
                rOffset = np.array([x, y])
                neighbor_search(pq = self.pq, root = root, particles = particles, r = self.r, rOffset = rOffset)

    def conserve_bounds_periodic(self):

        self.r = np.where(self.r < 0, self.r + 100, np.where(self.r > 100, self.r - 100, self.r))

    def conserve_bounds_reflective(self):
        # has yet to be implemented used, but is implemented

        self.r = np.where(self.r < 0, - self.r , np.where(self.r > 100, 100 - (self.r - 100), self.r))


    def top_hat_kernel(r, h):

        return np.where(np.logical_and(0 < (r / h), (r / h) < 1),
                        1 / (np.pi * h ** 2),
                        0)

    def monahan_kernel(r, h, s=40 / 7 * np.pi, d=2):

        w = s / h ** d
        w1 = np.zeros_like(r)

        mask1 = np.where(np.logical_and(0 <= (r / h), (r / h) < 0.5))
        w1[mask1] = 6 * (r[mask1] / h[mask1]) ** 3 - 6 * (r[mask1] / h[mask1]) ** 2 + 1

        mask2 = np.where(np.logical_and(0.5 <= (r / h), (r / h) <= 1))
        w1[mask2] = 2 * (1 - (r[mask2] / h[mask2])) ** 3

        return w * w1

    def calc_density(self, kernel = monahan_kernel):

        heap = np.array(self.pq.heap, dtype = object)
        masses = heap[:, 3]
        r = (-heap[:, 0]) ** 0.5
        h = np.ones_like(r) * self.pq.key() ** 0.5

        self.rho = (masses * kernel(r, h)).sum()

    def calc_initenergy(self):

        self.e = self.P / self.rho * (self.gamma - 1)
        self.epred = self.e

    def calc_csound(self):

        self.c = (self.gamma * (self.gamma - 1) * self.epred) ** 0.5

    def calc_pressure(self):

        self.P = self.epred * self.rho * (self.gamma - 1)


    def nabla_monahan_kernel(r, h, s=40 / 7 * np.pi, d=2):
        w = s / h ** d
        w1 = np.zeros_like(r)

        mask1 = np.where(np.logical_and(0 <= (r / h), (r / h) < 0.5)) # only apply to cases where not particle itself, exclude r == 0
        w1[mask1] = 18 * (r[mask1] / h[mask1]) ** 2 - 12 * (r[mask1] / h[mask1])

        mask2 = np.where(np.logical_and(0.5 <= (r / h), (r / h) <= 1))
        w1[mask2] = 2 * (1 - (r[mask2] / h[mask2])) ** 3

        return w * w1

    def nabla_density(self, kernel = nabla_monahan_kernel):

        heap = np.array(self.pq.heap, dtype = object)
        # p_r = np.ones_like(heap[:, 0])
        # p_r[:, 0] = self.r[0]
        # p_r[:, 1] = self.r[1]
        nn_rx = np.array([p.r[0] for p in heap[:, 1]])
        nn_ry = np.array([p.r[1] for p in heap[:, 1]])
        nn_h = np.array([p.pq.key() ** 0.5 for p in heap[:, 1]])
        
        r = np.absolute((-heap[:, 0]) ** 0.5) # is this correct

        # minumum of all possible cases to respect boundary conditions
        rx = np.array([list(np.absolute(nn_rx - self.r[0])),
                      list(100 - nn_rx + self.r[0]),
                      list(100 - self.r[0] + nn_rx)]).min(axis = 0)

        ry = np.array([list(np.absolute(nn_ry - self.r[1])),
                      list(100 - nn_ry + self.r[1]),
                      list(100 - self.r[1] + nn_ry)]).min(axis = 0)

        # the maximum radius
        h = np.ones_like(r) * self.pq.key() ** 0.5
        # mean max distance
        hab = (h + nn_h) * 0.5


        dx, dy = np.zeros(len(r)), np.zeros(len(r))

        mask = np.where(r > 0)

        dx[mask] = kernel(r[mask], h[mask]) * (1 / r[mask] * rx[mask])
        dy[mask] = kernel(r[mask], h[mask]) * (1 / r[mask] * ry[mask])


        return dx, dy, r, hab, mask


    def calc_dvdt_dedt(self, kernel = nabla_monahan_kernel):

        # extract info from neighbours to array for faster compiling
        nn_values = np.array([[p[1].P, p[1].rho, p[1].m, p[1].c] for p in self.pq.heap])
        nn_vpred = np.array([p[1].vpred for p in self.pq.heap])

        # respect boundary conditions
        # r_ab_x = np.array([np.absolute(p[1].r - self.r) for p in self.pq.heap])
        # r_ab_y = np.array([np.absolute(p[1].r - self.r) for p in self.pq.heap])
        # todo continue working here next, respect boundary conditions
        r_ab = np.array([np.absolute(p[1].r - self.r) for p in self.pq.heap])
        r_ab[:, 0] = np.where(r_ab[:, 0] < 0, r_ab[:, 0] + 100, np.where(r_ab[:, 0] > 100, r_ab[:, 0] - 100, r_ab[:, 0]))
        r_ab[:, 1] = np.where(r_ab[:, 1] < 0, r_ab[:, 1] + 100, np.where(r_ab[:, 1] > 100, r_ab[:, 1] - 100, r_ab[:, 1]))
        # print(r_ab)

        ones = np.ones(len(nn_values))
        p_rho = ones * self.rho
        p_Pa = ones * self.P
        p_vpred = np.ones((len(nn_values), 2))
        p_vpred[:, 0] *= self.vpred[0]
        p_vpred[:, 1] *= self.vpred[1]
        # obtain nabla kernel density and values for artificial velocity
        nabla_x, nabla_y, distr_ab, h_ab, mask = self.nabla_density(kernel = kernel)
        rho_ab = (nn_values[:, 1] + self.rho) * 0.5
        c_ab = (nn_values[:, 3] + self.c) * 0.5

        nabla_e = np.ones((len(nn_values), 2))
        nabla_e[:, 0] = nabla_x
        nabla_e[:, 1] = nabla_y
        delta_vpred = p_vpred - nn_vpred
        # print(f"vab {delta_vpred}")
        # print(f"rab {r_ab}")
        # print(f"hab {h_ab}")
        # print(f"rhoab {rho_ab}")
        # print(f"cab {c_ab}")

        self.artificial_viscosity(delta_vpred, distr_ab, r_ab, h_ab, rho_ab, c_ab, mask)

        c_dv = nn_values[:, 2] * ((p_Pa / p_rho ** 2) + (nn_values[:, 0] / nn_values[:, 1] ** 2) + self.pi_ab)  # dv constant in eq

        c_de = nn_values[:, 2] * np.array(
            [np.dot(delta_vpred[i], nabla_e[i]) for i in range(len(nabla_e))])  # ev constant in eq.

        # dv
        dvx = - (c_dv * nabla_x).sum()
        dvy = - (c_dv * nabla_y).sum()
        # de
        de = (self.P / self.rho ** 2) * c_de.sum()

        self.dv = np.array([dvx, dvy])
        self.de = de

    def artificial_viscosity(self, v_ab, distr_ab, r_ab, h_ab, rho_ab, c_ab, mask):

        alpha = 1
        beta = 2

        dot_vxr_ab = np.array([np.dot(v_ab[i], r_ab[i]) for i in range(len(v_ab))])

        nu_ab = np.zeros(len(distr_ab))
        nu_ab = (h_ab * dot_vxr_ab) / (distr_ab ** 2 + 1e-3) # 1e-6 to avoid explosion
        print(f"nu {nu_ab}")


        pi_ab = np.where(dot_vxr_ab > 0, (-alpha * c_ab * nu_ab + beta * nu_ab ** 2) / rho_ab, 0.)
        print(f"pi {pi_ab}")

        self.pi_ab = pi_ab




class cell:

    def __init__(self, rLow, rHigh, lower, upper):
        self.rLow = rLow  # [xMin, yMin]
        self.rHigh = rHigh  # [yMax, yMax]
        self.iLower = lower  # index to first particle in particle array
        self.iUpper = upper  # index to last particle in particle array
        self.pLower = None  # reference to tree cell for lower part
        self.pUpper = None  # reference to tree cell for upper part
        self.rc = self.rLow + self.rHigh / 2

    def celldist2(self, r):
        """Calculates the squared minimum distance between a particle
        position and this node."""
        d1 = r - self.rHigh
        d2 = self.rLow - r
        d1 = np.maximum(d1, d2)
        d1 = np.maximum(d1, np.zeros_like(d1))
        return d1.dot(d1)

class pq:
    def __init__(self, N):
        self.heap = []
        sentinel = (-np.inf, None, np.zeros(2))
        for i in range(N):
            heappush(self.heap, sentinel)

    def key(self):
        """
        :return: max distance of heap
        """
        return -1 * self.heap[0][0]

    def replace(self, d2, j, r, m):
        item = (-d2, j, r, m)
        # print(item)
        heapreplace(self.heap, item)

def partition(array, i, j, v, d):

    l = i
    if len(array) == 0:
        return l
    for k in range(i, j+1):
        if array[k].r[d] < v:
            array[k], array[l] = array[l], array[k]
            l += 1
    return l

def treebuild(A, root, dim):

    v = 0.5 * (root.rLow[dim] + root.rHigh[dim])
    s = partition(A, root.iLower, root.iUpper, v, dim)

    # may have two parts: lower..s-1 and s..upper
    if s > root.iLower:  # there is a lower part:

        rLow, rHigh = root.rLow.copy(), root.rHigh.copy()
        rHigh[dim] = v
        cLow = cell(rLow=rLow,
                    rHigh=rHigh,
                    lower=root.iLower,
                    upper=s - 1)
        root.pLower = cLow
        if len(A[root.iLower:s]) > 8:  # there are more than 8 particles in cell:
            treebuild(A, cLow, 1 - dim)
    if s <= root.iUpper:

        rLow, rHigh = root.rLow.copy(), root.rHigh.copy()
        rLow[dim] = v
        cHigh = cell(rLow=rLow,
                     rHigh=rHigh,
                     lower=s,
                     upper=root.iUpper)
        root.pUpper = cHigh
        if len(A[s:root.iUpper + 1]) > 8:
            treebuild(A, cHigh, 1 - dim)


    return A, root, dim

def plottree(root):
    # draw a rectangle specified by rLow and rHigh
    if root.pLower:
        plottree(root.pLower)
    if root.pUpper:
        plottree(root.pUpper)
    xl = root.rLow[0]
    xh = root.rHigh[0]
    yl = root.rLow[1]
    yh = root.rHigh[1]
    plt.plot([xl,xh],[yl,yl], color="k")
    plt.plot([xl,xh],[yh,yh], color="k")
    plt.plot([xl,xl],[yl,yh], color="k")
    plt.plot([xh,xh],[yl,yh], color="k")

def random_particles(num, lowc=0, highc=100, lowm=0, highm=1, P = 1):
    """
    :param num: number of particles generated
    :param low: low bound of square coords
    :param high:  high bound of square coords
    :return: list of particles
    """
    A = []

    for coord, mass in zip(np.random.uniform(low=lowc, high=highc, size=(num, 2)), np.random.uniform(low=lowm, high=highm, size=(num))):
        A.append(particle(r=coord, m = mass, P = P))

    return A


def neighbor_search_periodic(pq, root, particles, r, period):
    # walk the closest image first (at offset=[0, 0])
    for y in [0.0, -period[1], period[1]]:
        for x in [0.0, -period[0], period[0]]:
            rOffset = np.array([x, y])
            neighbor_search(pq, root, particles, r, rOffset)


def neighbor_search(pq, root, particles, r, rOffset):
    """Do a nearest neighbor search for particle at  'r' in the tree 'root'
       using the priority queue 'pq'. 'rOffset' is the offset of the root
       node from unit cell, used for periodic boundaries.
       'particles' is the array of all particles.
    """
    if root is None:
        return

    ri = r + rOffset
    # print(ri)
    if root.pLower is not None and root.pUpper is not None:
        d2_lower = root.pLower.celldist2(ri) # dist2(root.pLower.rc, ri) # dist2 is some other funcion ig
        d2_upper = root.pLower.celldist2(ri) # dist2(root.pUpper.rc, ri)
        if d2_lower <= d2_upper:
            if root.pLower.celldist2(ri) < pq.key():
                neighbor_search(pq, root.pLower, particles, r, rOffset)
            if root.pUpper.celldist2(ri) < pq.key():
                neighbor_search(pq, root.pUpper, particles, r, rOffset)
        else:
            if root.pUpper.celldist2(ri) < pq.key():
                neighbor_search(pq, root.pUpper, particles, r, rOffset)
            if root.pLower.celldist2(ri) < pq.key():
                neighbor_search(pq, root.pLower, particles, r, rOffset)
    elif root.pLower is not None:
        neighbor_search(pq, root.pLower, particles, r, rOffset)
    elif root.pUpper is not None:
        neighbor_search(pq, root.pUpper, particles, r, rOffset)
    else:  # root is a leaf cell
        for j in range(root.iLower, root.iUpper + 1):
            d2 = dist2(particles[j].r, ri)
            if d2 < pq.key():
                pq.replace(d2, particles[j], particles[j].r - rOffset, particles[j].m)



def dist2(p, q):

    return np.sum((q-p) ** 2)

def monahan_kernel(r, h, s=40 / 7 * np.pi, d=2):

    w = s / h ** d
    w1 = np.zeros_like(r)

    mask1 = np.where(np.logical_and(0 <= (r / h), (r / h) < 0.5))
    w1[mask1] = 6 * (r[mask1] / h[mask1]) ** 3 - 6 * (r[mask1] / h[mask1]) ** 2 + 1

    mask2 = np.where(np.logical_and(0.5 <= (r / h), (r / h) <= 1))
    w1[mask2] = 2 * (1 - (r[mask2] / h[mask2])) ** 3

    return w * w1

def nabla_monahan_kernel(r, h, s=40 / 7 * np.pi, d=2):
    w = s / h ** d
    w1 = np.zeros_like(r)

    mask1 = np.where(np.logical_and(0 <= (r / h), (r / h) < 0.5)) # only apply to cases where not particle itself, exclude r == 0
    w1[mask1] = 18 * (r[mask1] / h[mask1]) ** 2 - 12 * (r[mask1] / h[mask1])

    mask2 = np.where(np.logical_and(0.5 <= (r / h), (r / h) <= 1))
    w1[mask2] = 2 * (1 - (r[mask2] / h[mask2])) ** 3

    return w * w1

def wendtland_kernel(rab, rm):

    h = rm / 2
    q = np.absolute(rab) / h
    a2d = 7 / 4 * np.pi * h ** 2

    fq = np.where(np.logical_and(0 <= q, q <= 2), (1 - q/2) ** 4 * (1 + 2 * q), 0.0)

    return a2d * fq

def nabla_wendtland_kernel(rab, rm):

    h = rm / 2
    q = np.absolute(rab) / h
    a2d = 7 / 4 * np.pi * h ** 2

    fqprime = np.where(np.logical_and(0 <= q, q <= 2), -5 * q * (1 - q/2)**3, 0.0)

    return a2d / h * fqprime




def SPH_step(A, dt, nabla_kernel):

    # root = cell(rLow=np.array([0, 0]),
    #             rHigh=np.array([100, 100]),
    #             lower=0,
    #             upper=len(A) - 1)
    ## Drift 1
    for p in A:
        # update r with the previous v on dt/2
        p.r += p.v * (dt / 2)
        # avoid particle out of bounds and conserve periodic bounds
        p.conserve_bounds_periodic()
    for p in A:
        # predict velocity from de and dv from last iteration
        p.epred = p.e + p.de * (dt/2)
        p.vpred = p.v + p.dv * (dt/2)
    # for p in A:
        # calculate the speed of sound from epred
        p.calc_csound()
        # it is important to reinitialize the pq, otherwise the particle itself accumulates in pq with every iteration
        p.reset_pq()


    # recalculate the interactions with treebuild
    treebuild(A, root, 0)
    # recalculate density
    for p in A:
        p.neighbor_search_periodic(root=root, particles=A, period=np.array([100., 100.]))
        p.calc_density()

    ## Kick
    # calculate the real pressure from epred
    for p in A:
        p.calc_pressure()
    # calulate the acceleration and denergy
    # for p in A:
        p.calc_dvdt_dedt(kernel = nabla_kernel)
        # print(f"D/DT Kick: dv:{p.dv}, de {p.de}") derivatives seem okay
    # update energy and velocity
    for p in A:
        #print(f"Before Kick: v:{p.v}, de {p.e}")
        p.v += p.dv * dt
        p.e += p.de * dt
        #print(f"D/DT Kick: dv:{p.dv}, de {p.de}")
        #print(f"After Kick: v:{p.v}, de {p.e}")
    ## Drift 2
    # update particle positions
    for p in A:
        p.r += p.v * (dt/2)
        # conserve periodic bounds
        p.conserve_bounds_periodic()
        print(p.r)



def SPH_gather_scatter_step(A, dt):
    pass


# SPH

# first initialize the particles
num_particles = 100
A = random_particles(num=num_particles, lowc=0, highc=99, lowm=0.99, highm=1.)

root = cell(rLow=np.array([0, 0]),
            rHigh=np.array([100, 100]),
            lower=0,
            upper=len(A) - 1)

treebuild(A, root, 0)

steps = 100
dt = 0.001


# intitializing the initial particles
for p in A:
    # p.pq = pq(N=N)
    p.neighbor_search_periodic(root=root, particles=A, period=np.array([100., 100.]))
    p.calc_density(kernel=monahan_kernel)
A[0].P = 10
for p in A:
    p.calc_initenergy()

# SPH_step(A, dt, nabla_monahan_kernel)

max_time = 1
t = 0
print("Finished particle initialization, starting animation rendering:")
dt = 1e-10
while t < max_time:
    t += dt
    SPH_step(A, dt, nabla_monahan_kernel)
    dt = max([p.rho for p in A]) / max([p.c for p in A]) # that dt doesnt get to big!
    print(f"dt: {dt}")

# important: is it an issue that kernel is always zero at the furthest neighbour?, means that we always have to particles with nabla = 0

# animation of the simulation
fig = plt.figure()
ax = plt.axes()
#plt.colorbar()
import progressbar
from datetime import datetime
import time

# use example of progressbars
'''
import time
import progressbar
bar = progressbar.ProgressBar()
for i in bar(range(100)):
    time.sleep(0.02)
'''


# bar = progressbar.ProgressBar()
# bar.max_value = steps
#
#
#
# def animate(i):
#
#     global A, dt
#     SPH_step(A, dt, nabla_monahan_kernel)
#     ax.clear()
#     result = np.array([[p.r[0], p.r[1], p.rho] for p in A])
#     ax.scatter(result[:, 0], result[:, 1], s= 10, c=result[:,2])
#     bar.update(i+1)
#     # print(f"step: {i}")
#     #plt.colorbar()
#
# anim = animation.FuncAnimation(fig, animate, frames=steps, interval=200)
#
# # if time stamp from time.time(): use datetime.fromtimestamp(time.time()).strftime(...)
# sim_date = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
#
# anim.save(f"SPH_{num_particles}particles_{steps}steps_date{sim_date}.mp4", fps=30)



# print("General observation: velocity accumulates depending on position in A!")
# print("Dont forget NN != 32 on top!")
# for p in A:
#     print(p.v)
# print("All particles have the same speed?")



if __name__ == "__main__":
    pass
    #max(rho) / max(csound)