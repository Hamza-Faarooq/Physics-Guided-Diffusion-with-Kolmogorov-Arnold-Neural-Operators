import numpy as np
import matplotlib.pyplot as plt


def plot_turbulence(flow):

    u = flow[0]

    v = flow[1]

    dvdx = np.gradient(v, axis=1)

    dudy = np.gradient(u, axis=0)

    vort = dvdx - dudy

    plt.figure(figsize=(6,6))

    plt.imshow(
        vort,
        cmap="RdBu"
    )

    plt.title("Turbulent Vorticity")

    plt.colorbar()

    plt.show()
