


import numpy as np
import matplotlib.pyplot as plt

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

    def celldist2(self, r):
        """Calculates the squared minimum distance between a particle
        position and this node."""
        d1 = r - self.rHigh
        d2 = self.rLow - r
        d1 = np.maximum(d1, d2)
        d1 = np.maximum(d1, np.zeros_like(d1))
        return d1.dot(d1)

def partition(array, i, j, v, d):

    l = i
    if len(array) == 0:
        return l
    for k in range(i, j+1):
        if array[k].r[d] < v:
            array[k], array[l] = array[l], array[k]
            l += 1
    return l


def test1():
    """
    partition in middle
    """
    A = np.array([particle(np.array([1, 0])), particle(np.array([2, 0])), particle(np.array([3, 0])),
                  particle(np.array([4, 0]))])
    s = partition(A, 0, 4, 3, 0)
    # print(s)
    if s == 2:
        return True
    return False


def test2():
    """
    partition empty
    """
    A = np.array([])
    s = partition(A, 0, 0, 1, 0)
    # print(s)
    if s == 0:
        return True
    return False


def test3():
    """flipped array"""
    A = np.array([particle(np.array([4, 0])), particle(np.array([3, 0])), particle(np.array([2, 0])),
                  particle(np.array([1, 0]))])
    s = partition(A, 0, 4, 3, 0)
    # print(s)
    if s == 2:
        return True
    return False


def test4():
    """boundary test left"""
    A = np.array([particle(np.array([4, 0])), particle(np.array([5, 0])), particle(np.array([6, 0])),
                  particle(np.array([7, 0]))])
    s = partition(A, 0, 4, 3, 0)
    # print(s)
    if s == 0:
        return True
    return False


def test5():
    """boundary test right"""
    A = np.array([particle(np.array([4, 0])), particle(np.array([5, 0])), particle(np.array([6, 0])),
                  particle(np.array([7, 0]))])
    s = partition(A, 0, 4, 9, 0)
    # print(s)
    if s == 4:
        return True
    return False


def testAll():
    if test1() == False:
        return False

    if test2() == False:
        return False

    if test3() == False:
        return False

    if test4() == False:
        return False

    if test5() == False:
        return False

    return True


def treebuild(A, root, dim):

    v = 0.5 * (root.rLow[dim] + root.rHigh[dim])
    s = partition(A, root.iLower, root.iUpper, v, dim)
    print("v", v)
    print("s", s)
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
        if len(A[root.iLower:s]):  # there are more than 8 particles in cell:
            print("Upper cell:", len(A[root.iLower:root.iUpper]))
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

if __name__ == "__main__":

    A = random_particles(num = 100, low = 0, high = 99)

    root = cell(rLow = [0, 0],
                rHigh = [100, 100],
                lower = 0,
                upper = len(A) - 1)

    treebuild(A,root,0)

    for ele in A:
        plt.scatter(ele.r[0],ele.r[1], color = "red")
    plottree(root)

    plt.show()