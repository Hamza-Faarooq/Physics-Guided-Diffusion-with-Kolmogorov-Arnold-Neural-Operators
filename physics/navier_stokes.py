import torch


def gradients(y, x):

    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]


def navier_stokes(model, coords):

    coords.requires_grad_(True)

    pred = model(coords.unsqueeze(0)).squeeze(0)

    u = pred[:, 0:1]
    v = pred[:, 1:2]
    p = pred[:, 2:3]

    grads_u = gradients(u, coords)

    u_x = grads_u[:, 0:1]
    u_y = grads_u[:, 1:2]
    u_t = grads_u[:, 2:3]

    grads_v = gradients(v, coords)

    v_x = grads_v[:, 0:1]
    v_y = grads_v[:, 1:2]
    v_t = grads_v[:, 2:3]

    grads_p = gradients(p, coords)

    p_x = grads_p[:, 0:1]
    p_y = grads_p[:, 1:2]

    u_xx = gradients(u_x, coords)[:, 0:1]
    u_yy = gradients(u_y, coords)[:, 1:2]

    v_xx = gradients(v_x, coords)[:, 0:1]
    v_yy = gradients(v_y, coords)[:, 1:2]

    return (
        u, v, p,
        u_x, u_y,
        v_x, v_y,
        u_t, v_t,
        p_x, p_y,
        u_xx, u_yy,
        v_xx, v_yy
    )
