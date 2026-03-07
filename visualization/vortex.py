import numpy as np
import matplotlib.pyplot as plt


def plot_vorticity(u, v):

    dvdx = np.gradient(v, axis=1)

    dudy = np.gradient(u, axis=0)

    vorticity = dvdx - dudy

    plt.figure(figsize=(6,6))

    plt.imshow(
        vorticity,
        cmap="RdBu"
    )

    plt.colorbar()

    plt.title("Vorticity Field")

    plt.show()
