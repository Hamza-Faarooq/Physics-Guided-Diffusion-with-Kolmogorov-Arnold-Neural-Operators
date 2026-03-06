import torch
import torch.nn as nn
from kan import KAN
from models.spectral_conv import SpectralConv2d


class KANFNOBlock(nn.Module):

    def __init__(self, width):

        super().__init__()

        self.spectral = SpectralConv2d(width, width, 16, 16)

        self.conv = nn.Conv2d(width, width, 1)

        self.kan = KAN(width=[width, width, width])

        self.activation = nn.GELU()

    def forward(self, x):

        x1 = self.spectral(x)

        x2 = self.conv(x)

        x = x1 + x2

        b, c, h, w = x.shape

        x_flat = x.permute(0, 2, 3, 1).reshape(-1, c)

        x_flat = self.kan(x_flat)

        x = x_flat.reshape(b, h, w, c).permute(0, 3, 1, 2)

        return self.activation(x)
