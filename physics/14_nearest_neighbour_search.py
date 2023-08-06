from heapq import *
import numpy as np
import matplotlib.pyplot as plt
import math


class particle:
    def __init__(self, r):
        self.r = r  # positionn of the particle [x, y]
        self.rho = 0.0  # density of the particle
        # ...  more properties of the particle

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
        sentinel = (-np.inf, np.nan, np.zeros(2))
        for i in range(N):
            heappush(self.heap, sentinel)

    def key(self):
        """
        :return: max distance of heap
        """
        return -1 * self.heap[0][0]

    def replace(self, d2, j, r):
        item = (-d2, j, r)
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
    # print("v", v)
    # print("s", s)
    # print(s)

    # may have two parts: lower..s-1 and s..upper
    if s > root.iLower:  # there is a lower part: # todo study algo to find out which bound to change, as only one changes
        # rLow[dim] = root.rLow[dim]
        # rLow[dim - 1] = root.rLow[dim-1]
        # rHigh[dim] = root.rLow[dim] + root.rHigh[dim] / 2
        # rHigh[dim - 1] = root.rHigh[dim-1]
        rLow, rHigh = root.rLow.copy(), root.rHigh.copy()
        rHigh[dim] = v
        cLow = cell(rLow=rLow,
                    rHigh=rHigh,
                    lower=root.iLower,
                    upper=s - 1)
        root.pLower = cLow
        if len(A[root.iLower:s]) > 8:  # there are more than 8 particles in cell:
            # print("Upper cell:", len(A[root.iLower:root.iUpper]))
            treebuild(A, cLow, 1 - dim)
    if s <= root.iUpper:  # todo double check thought
        # rLow[dim] = root.rLow[dim] + root.rHigh[dim] / 2
        # rLow[dim - 1] = root.rLow[dim - 1]
        # rHigh[dim] = root.rHigh[dim]
        # rHigh[dim - 1] = root.rHigh[dim - 1]
        rLow, rHigh = root.rLow.copy(), root.rHigh.copy()
        rLow[dim] = v
        cHigh = cell(rLow=rLow,
                     rHigh=rHigh,
                     lower=s,
                     upper=root.iUpper)
        root.pUpper = cHigh
        if len(A[s:root.iUpper + 1]) > 8:
            treebuild(A, cHigh, 1 - dim)
            # print("Upper cell:", len(A[s:root.iUpper + 1]) > 8)
    # grafical representation of tree

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

def random_particles(num, low=0, high=1):
    """
    :param num: number of particles generated
    :param low: low bound of square coords
    :param high:  high bound of square coords
    :return: list of particles
    """
    A = []

    for coord in np.random.uniform(low=low, high=high, size=(num, 2)):
        A.append(particle(r=coord))

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
                pq.replace(d2, j, particles[j].r - rOffset)
    # for pq write a wrapper class that implements key() and replace() using heapq package



def dist2(p, q): # todo problem: same particle is added mutliple times, distance is computed differently for different times
    # print(p, q)
    return np.sum((q-p) ** 2)

def plotsearch(A, root, pq):
    pass



if __name__ == "__main__":
    A = random_particles(num = 100, low = 0, high = 99)

    root = cell(rLow = np.array([0, 0]),
                rHigh = np.array([100, 100]),
                lower = 0,
                upper = len(A) - 1)

    treebuild(A,root,0)

    pq = pq(N = 32)
    r = A[50].r

    # neighbor_search(pq=pq,
    #                 root = root,
    #                 particles=A,
    #                 r = r,
    #                 rOffset=np.array([0, 0]))

    neighbor_search_periodic(pq = pq,
                             root = root,
                             particles = A,
                             r = r,
                             period = np.array([100., 100.]))

    i_neighbours = sorted([ele[1] for ele in pq.heap])


    fig, ax = plt.subplots(1, 1, figsize = (10, 10))

    farest_d = pq.key() ** 0.5
    rad = plt.Circle(xy = (r[0], r[1]), radius = farest_d, alpha = 0.2, color = "green")
    ax.add_patch(rad)

    for i, ele in enumerate(A):
        if math.dist(ele.r, r) == 0.:
            ax.scatter(ele.r[0], ele.r[1], color="red", s = 100)
        elif i in i_neighbours:
            ax.scatter(ele.r[0], ele.r[1], color="green")
        else:
            ax.scatter(ele.r[0],ele.r[1], color = "dimgrey")

    ax.scatter([], [], color="red", label="search centre")
    ax.scatter([], [], color="green", label="neighbours")
    ax.scatter([], [], color="dimgrey", label="strangers")

    plottree(root)
    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    plt.show()

    # todo: problem the same is added multiple times to heap




