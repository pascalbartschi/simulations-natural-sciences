import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialization of global variables

u = 1
dt = 0.1
dx = 1
CFL = abs(u) < dx / dt  # pyhsical velocity must be smaller than grid velocity

if CFL:
    C = (dt * u) / dx
else:
    raise ValueError("Physical velocity (u) must be smaller than grid velocity (dx/dt)!")


def initialize_rho(N=400, N_marked=10):
    # rho array
    N_marked = int(N_marked * 0.5)
    rho = np.zeros(N)
    rho[(int(N / 2) - N_marked): (int(N / 2) + N_marked)] = 1

    return rho


def timestep(rho):
    rhoJup = np.roll(rho, -1)  # -1 for step to right
    rhoJdown = np.roll(rho, 1)

    return rhoJdown, rhoJup


def LAX_method(rho):
    # update down- and upstream rho
    rho_down, rho_up = timestep(rho)

    # LAX update
    rho_new = (0.5 * (rho_up + rho_down)) - (0.5 * C * (rho_up - rho_down))

    return rho_new


if u > 0:

    def CIR_upwind(rho):
        # update down- and upstream rho
        rho_down, rho_up = timestep(rho)

        # CIR
        rho_new = rho - (C * (rho - rho_down))

        return rho_new


    def LAX_WEN_upwind(rho):
        # update down- and upstream rho
        rho_down, rho_up = timestep(rho)

        # LAX WEN
        rho_new = (0.5 * C * (1 + C) * rho_down) + ((1 - C ** 2) * rho) - (0.5 * C * (1 - C) * rho_up)

        return rho_new



else:

    def CIR_upwind(rho):
        # update upstream rho
        rho_down, rho_up = timestep(rho)

        # CIR
        rho_new = rho - (C * (rho_up - rho))

        return rho_new


def solver(len_grid, len_rho, method, N_timesteps):
    rho = initialize_rho(N=len_grid, N_marked=len_rho)
    time = np.arange(0, N_timesteps + 1) * dt

    storage = np.zeros(shape=(len(rho), len(time)))
    storage[:, 0] = rho

    for i, t in enumerate(time[1:]):
        rho_new = method(rho)

        storage[:, i + 1] = rho_new.copy()

        rho = rho_new

    return storage, time


def animate_simulation(rounds, init_tuple, method):
    len_grid, len_rho = init_tuple

    N_steps = int(rounds * len_grid * (1 / dt))  # array will go around exactly one time for rounds = 1

    storage, time = solver(len_grid=len_grid, len_rho=len_rho, method=method, N_timesteps=N_steps)

    # plt.plot(range(len_grid), LAX_storage[:, 1])

    # animate a storage

    fig = plt.figure()  # figure object
    ax = plt.axes()  # single axis in figure
    ax.set_title(method.__name__)
    line, = ax.plot([], [], lw=2, color="purple")  # artist to store lines in, for blit argument

    
    if method.__name__ == "LAX_WEN_upwind":
        
        annotation = ax.annotate("Timestep: 0", xy=(5, 2))
        annotation.set_animated(True)
        ax.set_xlim(0, len_grid)
        ax.set_ylim(-0.1, 2.1)
    
    else:
        
        ax.set_xlim(0, len_grid)
        ax.set_ylim(-0.1, 1.1)
        annotation = ax.annotate("Timestep: 0", xy=(5, 1))  # annotation for timestep counter
        annotation.set_animated(True)

    # animation function.  This is called sequentially
    def animate(i):
        # x.clear()
        x = np.arange(len_grid)
        y = storage[:, i]
        line.set_data(x, y)

        # ax.clear()
        # annotation.set_xy = (0.02, 0.99)
        # annotation.set_("subfigure fraction")
        annotation.set_text("Timestep:" + str(time[i]))

        return line, annotation

    frames = np.array(list(range(0, N_steps // 100, 5)) + list(range(N_steps // 400, N_steps, 25)))

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = FuncAnimation(fig, animate, interval=20, frames=frames, blit=True)

    anim.save(method.__name__ + "_animation.mp4", fps=30, dpi=300, extra_args=['-vcodec', 'libx264'])

    return storage


def main():
    init = (400, 20)
    animate_simulation(1, init, LAX_method)
    animate_simulation(1, init, CIR_upwind)
    animate_simulation(5, init, LAX_WEN_upwind)
    

if __name__ == "__main__":
    main() # attention: takes approximately 5 min to run on my machine