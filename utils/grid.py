import torch
import numpy as np


def build_grid(resolution=128, device="cpu"):

    xs = np.linspace(0,1,resolution)
    ys = np.linspace(0,1,resolution)

    coords = []

    for x in xs:
        for y in ys:

            coords.append([x,y,0.0,0.5])

    coords = torch.tensor(coords).float().to(device)

    return coords
