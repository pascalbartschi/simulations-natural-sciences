import numpy as np
import matplotlib.pyplot as plt
import math



class vertice:
    def __init__(self, pos, gradient):
        self.pos = pos
        self.gradient = gradient


class cell:
    def __init__(self, v, pos):
        self.v = v
        self.pos = pos  # position of the vertex of interest to move
        self.A = None
        self.perimeter = None
        self.calc_area()
        self.calc_perimeter()

    def __name__(self):
        return "cell"

    def calc_area(self):
        """
        calculation of polygon area in counter-clockwise direction
        :param v: array of vertices
        :return: area of cell
        """
        # A = 0 # todo do it with slicing
        # for i in range(len(self.v)-1):
        #     A += 0.5 * (self.v[i, 0] * self.v[i+1, 1] - self.v[i+1, 0] * self.v[i, 1])

        A = -0.5 * (self.v[:, 0] * np.roll(self.v[:, 1], 1) - np.roll(self.v[:, 0], 1) * self.v[:, 1]).sum()

        self.A = A

    def calc_perimeter(self):

        #  self.surface = np.linalg.norm(self.v) # todo not sure whether this is correct
        self.perimeter = sum([math.dist(self.v[i-1], self.v[i]) for i in range(len(self.v))])


class tissue:

    def __init__(self, v, cells, params):
        A0, L, R, K = params
        self.cells = cells
        self.vertices = v
        self.A0 = A0
        self.Lambda = L
        self.Rho = R
        self.K = K
        self.edgeloc = [[0, 1], [1, 2],  [2, 3], [3, 4], [4, 8], [8, 9], [9, 10], [10, 7], [7, 6], [6, 0], [6, 5], [5, 2], [5, 8]]
        self.edges = None
        self.gradient = None
        self.totE = None
        self.calc_totE()

    def calc_edges(self):
        # self.edges = np.linalg.norm(self.vertices, axis = 1).sum()
        # self.edges = sum([math.dist(self.vertices[i - 1], self.vertices[i]) for i in range(len(self.vertices))])
        self.edges = np.array([math.dist(self.vertices[loc[0]], self.vertices[loc[1]]) for loc in self.edgeloc])

    def calc_totE(self):
        area_elasticity = ((self.K / 2) * (np.array([cell.A for cell in self.cells]) - self.A0) ** 2).sum()
        line_tension = self.Lambda / 2 * np.array([cell.perimeter for cell in self.cells]).sum()# just perimeters / 2
        perimeter_elasticity = (self.Rho / 2 ) * (np.array([cell.perimeter for cell in self.cells])**2).sum()

        self.totE = area_elasticity + line_tension + perimeter_elasticity

    def calc_gradient(self):

        xy_gradient = np.zeros(2)
        for dim in [0, 1]:
            delta = 1e-5
            E1 = self.totE
            # cell0.v[cell0.pos, dim] += delta
            for cell in self.cells:
                cell.v[cell.pos, dim] += delta
                cell.calc_perimeter()
                cell.calc_area()
            self.calc_totE()
            E2 = self.totE
            g = (E2 - E1) / delta
            xy_gradient[dim] = g

        self.gradient = xy_gradient

# vertices
v0 = [0.8, 5.5]
v1 = [2.4, 7.0]
v2 = [4.1, 6.4]
v3 = [6.3, 7.0]
v4 = [7.6, 4.8]
v5 = [4.0, 5.4]
v6 = [1.9, 3.4]
v7 = [1.9, 2.2]
v8 = [6.3, 3.3]
v9 = [6.3, 2.2]
v10 = [3.7, 1.0]
v_all = [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10]

# cells
vertices0 = np.array([v6, v5, v2, v1, v0])
vertices1 = np.array([v5, v8, v4, v3, v2])
vertices2 = np.array([v10, v9, v8, v5, v6, v7])

# parameters
A0 = 10 # microm^2
L = 0.56 # J*microm^1
R = 0.004 # J * microm^2
K = 1 # J * microm^-4
params = (A0, L, R, K)

cell0 = cell(v = vertices0, pos = 1)
cell1 = cell(v = vertices1, pos = 0)
cell2 = cell(v = vertices2, pos = 3)
all_cells = [cell0, cell1, cell2]

system = tissue(v = v_all, cells = all_cells, params = params)

system.calc_gradient()





# debug function

def print_cells(system):

    for i, cell in enumerate(system.cells):
        print(f"cell{i} A: {cell.A}, peri: {cell.perimeter}")