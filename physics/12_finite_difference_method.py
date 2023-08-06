import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# global variable definition

Nx = 200
Ny = 200
dt = 0.1
dx = 1.0 / Nx
dy = 1.0 / Ny
sigma_x = 10 * dx
sigma_y = 10 * dy
x0 = 2 * sigma_x
y0 = 2 * sigma_y
Ca = 0.48
Cb = 0.48

# test for boundary conditions

if (Ca > 1 or Ca < 0) or (Cb > 1 or Cb < 0):
    raise ValueError("Boundary Condition not fulfilled. Please choose appropriate Ca and/or Cb.")


def initialize_rho():
    rho = np.zeros((Nx, Ny))
    for j in range(Nx):
        for l in range(Ny):
            x = j * dx
            y = l * dy
            rho[j, l] = np.exp(-(x - x0) ** 2 / (2 * sigma_x ** 2)
                               - (y - y0) ** 2 / (2 * sigma_y ** 2))

    return rho


def CTU_step(rho):
    rho_star = (1 - Ca) * rho + Ca * np.roll(rho, 1, axis=0)  # x-axis roll
    rho_new = (1 - Cb) * rho_star + Cb * np.roll(rho_star, 1, axis=1)  # y axis roll

    return rho_new


def CIR_upwind(rho):
    rho_new = rho - Ca * (rho - np.roll(rho, 1, axis=0)) - Cb * (rho - np.roll(rho, 1, axis=1))

    return rho_new


def solver(method, N_timesteps):
    rho = initialize_rho()
    time = np.arange(0, N_timesteps + 1) * dt

    storage = np.zeros(shape=(rho.shape[0], rho.shape[1], len(time)))
    storage[:, :, 0] = rho

    for i, t in enumerate(time[1:]):
        rho_new = method(rho)

        storage[:, :, i + 1] = rho_new.copy()

        rho = rho_new

    return storage, time


def animate_simulation(N_steps, method):

    N_steps = N_steps  # array will go around exactly one time for rounds = 1

    storage, time = solver(method=method, N_timesteps=N_steps)

    # animate a storage

    fig = plt.figure()  # figure object
    ax = plt.axes()  # single axis in figure
    ax.set_title(method.__name__)
    im = ax.imshow(storage[:, :, 0], origin = "lower")  # artist to store lines in, for blit argument

    annotation = ax.annotate("Timestep: 0", xy=(5, 190), c = "white")  # annotation for timestep counter
    annotation.set_animated(True)

    # ax.set_xlim(0, len_grid)
    # ax.set_ylim(-0.1, 1.1)

    # animation function.  This is called sequentially
    def animate(i):

        image = storage[:,:, i]
        im.set_array(image)

        # ax.clear()
        # annotation.set_xy = (0.02, 0.99)
        # annotation.set_("subfigure fraction")
        annotation.set_text("Timestep:" + str(round(time[i], 1)))

        return im, annotation

    frames = N_steps

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = FuncAnimation(fig, animate, interval=20, frames=frames, blit=True)

    anim.save(method.__name__ + "_animation.mp4", fps=60, dpi=300, extra_args=['-vcodec', 'libx264'])

    return storage



def main():
    
    animate_simulation(1500, CTU_step)
    animate_simulation(1500, CIR_upwind)
    
if __name__ == "__main__":
    main() # attention: takes approximately 5 min to run on my machine

# NOTE: ax.plot returns tuple and ax.imshow single object! -Y important to know for FuncAnimation
