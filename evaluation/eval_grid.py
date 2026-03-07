import torch
import numpy as np


def evaluate_model(model, resolution=128, device="cpu"):

    xs = np.linspace(0, 1, resolution)

    ys = np.linspace(0, 1, resolution)

    grid = []

    for x in xs:

        for y in ys:

            grid.append([x, y, 0.0, 0.5])

    grid = torch.tensor(grid).float().to(device)

    with torch.no_grad():

        pred = model(grid.unsqueeze(0)).squeeze(0)

    u = pred[:, 0].reshape(resolution, resolution)

    v = pred[:, 1].reshape(resolution, resolution)

    p = pred[:, 2].reshape(resolution, resolution)

    return u.cpu().numpy(), v.cpu().numpy(), p.cpu().numpy()
