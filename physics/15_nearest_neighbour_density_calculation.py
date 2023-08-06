from heapq import *
import numpy as np
import matplotlib.pyplot as plt
import math


class particle:
    def __init__(self, r, m = 0.0):
        self.r = r  # positionn of the particle [x, y]
        self.m = m
        self.rho = 0.0  # density of the particle
        self.pq = None

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
    if s > root.iLower:  # there is a lower part:
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
    if s <= root.iUpper:  
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

def random_particles(num, lowc=0, highc=100, lowm=0, highm=1):
    """
    :param num: number of particles generated
    :param low: low bound of square coords
    :param high:  high bound of square coords
    :return: list of particles
    """
    A = []

    for coord, mass in zip(np.random.uniform(low=lowc, high=highc, size=(num, 2)), np.random.uniform(low=lowm, high=highm, size=(num))):
        A.append(particle(r=coord, m = mass))

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
    # for pq write a wrapper class that implements key() and replace() using heapq package



def dist2(p, q):
    # print(p, q)
    return np.sum((q-p) ** 2)

def top_hat_kernel(r, h):

    return np.where(np.logical_and(0 < (r / h), (r / h) < 1),
                    1 / (np.pi * h ** 2),
                    0)

    # return (-np.array(particle.pq.heap)[:, 0]).sum() / 2 * np.pi * particle.pq.key() # distances of all the arrays
    # if 0 < (r / h) < 1:
    #     return 1 / (np.pi * h ** 2)
    # else:
    #     return 0


def monahan_kernel(r, h, s = 40 / 7 * np.pi, d = 2): # todo potenital numpy array with where?

    w = s / h ** d
    w1 = np.zeros_like(r)

    mask1 = np.where(np.logical_and(0 <= (r/h),(r/h) < 0.5))
    w1[mask1] = 6 * (r[mask1]/h[mask1])**3 - 6 * (r[mask1]/h[mask1])**2 + 1

    mask2 = np.where(np.logical_and(0.5 <= (r/h), (r/h) <= 1))
    w1[mask2] = 2 * (1 - (r[mask2]/h[mask2]))**3

    return w * w1

    # if 0 <= (r/h) < 0.5:
    #     w1 = 6 * (r/h)**3 - 6 * (r/h)**2 + 1
    #     return w1 * w
    #
    # elif 0.5 <= (r/h) <= 1:
    #     w1 = 2 * (1 - (r/h))**3
    #     return w1 * w
    #
    # else:
    #     return 0


def density(kernel, particle):

    rho = 0
    heap = np.array(particle.pq.heap)
    masses = heap[:, 3]
    r = (-heap[:, 0]) ** 0.5
    h = np.ones_like(r) * particle.pq.key() ** 0.5

    particle.rho = (masses * kernel(r, h)).sum()

    # for neighbour in np.array(particle.pq.heap)[:, 1]:
    #     rho += neighbour.m * kernel(dist2(particle.r, neighbour.r) ** 0.5, particle.pq.key() ** 0.5) # mass, distance between two points, and max dist
    #
    # particle.rho = rho


# todo: functions to plot knn search and grid of tree
if __name__ == "__main__":
    # tree with 100 particles
    A = random_particles(num=1000, lowc=0, highc=99, lowm=0.99, highm=1.)

    root = cell(rLow=np.array([0, 0]),
                rHigh=np.array([100, 100]),
                lower=0,
                upper=len(A) - 1)

    treebuild(A, root, 0)

    # compile neighbours for all particles
    N = 32
    for p in A:
        p.pq = pq(N=N)
        neighbor_search_periodic(pq=p.pq,
                                 root=root,
                                 particles=A,
                                 r=p.r,
                                 period=np.array([100., 100.]))
        density(kernel = monahan_kernel,
                particle = p)

    # plot
    result = np.array([[p.r[0], p.r[1], p.rho] for p in A])
    plt.scatter(result[:, 0], result[:, 1], s= 10, c=result[:,2])
    plt.colorbar()
    plt.show()


