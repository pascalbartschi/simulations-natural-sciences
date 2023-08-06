import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
cols = ["COL1", "COl2"]
rows = ["Dynamics", "Phase plots"]

plt.setp(axs.flat, xlabel="xlabel", ylabel="ylabel")
fig.subplots_adjust(right=0.95, left=0.15)

for ax, col in zip(axs[0], cols):
    ax.annotate(text=col, xy=(0.5, 1.1), annotation_clip=False, fontsize=15, ha='right', va='center')

for ax, row in zip(axs[:, 0], rows):
    ax.annotate(text=row, xy=(-0.4, 0.5), annotation_clip=False, fontsize=15)

# plt.show()

x = axs[0, 1].get_shared_x_axes()
