# import libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# planned sets
C_array = np.array([0.2 + 0.4j, -0.4 + 0.6j, -0.5 + 0.5j, -0.6 + 0.8j, 0.2 - 0.8j, 0.4 + 0.6j])

# iterate over sets
for C in C_array:
    x = np.linspace(-1.5, 1.5, 1000)
    y = np.linspace(-1.5, 1.5, 1000)
    X, Y = np.meshgrid(x, y)  ## mesh
    C = C  ## set C
    julia = X + 1j * Y  ## modify to complex mesh
    Z = np.copy(julia)  ## leaves julia 'unharmed'

    rmax = np.maximum(abs(Z), 2 * np.ones_like(Z))  ## max radius
    iterations = np.zeros_like(Z, dtype=int)  ## iter for colorcode

    maxiter = 100
    n = 0
    while n <= maxiter:
        cond = abs(Z) < rmax  ## condition
        Z[cond] = Z[cond] ** 2 + C  ## store in array
        iterations[cond] += 1  ## iter for colorcode
        n += 1  ## limit loop

    title = str(C)
    plt.scatter(julia.real, julia.imag, s=1, c=iterations, cmap='jet')  ## plot from storage array
    plt.title(title)

    # iterations = iterations / np.max(iterations)
    # cmap = cm.get_cmap(name='rainbow')
    # color = iterations
    # plt.scatter(C.real, C.imag, s=1, c=color, cmap=cmap)

    plt.savefig("julia_set_" + title + ".png")  ## save plot
    plt.show()
