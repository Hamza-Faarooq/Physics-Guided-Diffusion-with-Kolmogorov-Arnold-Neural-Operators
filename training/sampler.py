import torch


def sample_points(n, device="cpu"):

    x = torch.rand(n, 1, device=device)

    y = torch.rand(n, 1, device=device)

    t = torch.rand(n, 1, device=device)

    Re = torch.randint(50, 200, (n, 1), device=device).float() / 200.0

    coords = torch.cat([x, y, t, Re], dim=1)

    return coords
