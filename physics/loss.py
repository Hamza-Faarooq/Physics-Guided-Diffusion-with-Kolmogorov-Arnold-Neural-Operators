import torch
from physics.navier_stokes import navier_stokes


def physics_loss(model, coords, rho=1.0, mu=0.01):

    (
        u, v, p,
        u_x, u_y,
        v_x, v_y,
        u_t, v_t,
        p_x, p_y,
        u_xx, u_yy,
        v_xx, v_yy
    ) = navier_stokes(model, coords)

    momentum_u = (
        u_t
        + u * u_x
        + v * u_y
        + p_x / rho
        - mu * (u_xx + u_yy)
    )

    momentum_v = (
        v_t
        + u * v_x
        + v * v_y
        + p_y / rho
        - mu * (v_xx + v_yy)
    )

    continuity = u_x + v_y

    loss = (
        torch.mean(momentum_u ** 2)
        + torch.mean(momentum_v ** 2)
        + torch.mean(continuity ** 2)
    )

    return loss
