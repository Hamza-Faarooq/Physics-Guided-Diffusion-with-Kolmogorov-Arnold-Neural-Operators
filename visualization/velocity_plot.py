import numpy as np
import matplotlib.pyplot as plt


def plot_velocity(u, v):

    res = u.shape[0]

    x = np.linspace(0, 1, res)
    y = np.linspace(0, 1, res)

    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(6,6))

    plt.streamplot(
        X,
        Y,
        u,
        v,
        density=2
    )

    plt.title("Velocity Streamlines")

    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()
