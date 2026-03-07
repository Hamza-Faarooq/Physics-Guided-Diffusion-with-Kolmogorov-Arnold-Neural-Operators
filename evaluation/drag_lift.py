import numpy as np


def compute_drag_lift(u, v, p):

    pressure_force = np.sum(p)

    shear_force = np.sum(u * v)

    drag = pressure_force + shear_force

    lift = np.sum(v)

    return drag, lift
