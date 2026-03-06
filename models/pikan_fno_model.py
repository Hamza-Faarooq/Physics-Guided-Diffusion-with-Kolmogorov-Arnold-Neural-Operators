import torch
import torch.nn as nn
from models.kan_fno_block import KANFNOBlock


class PIKANFNO(nn.Module):

    def __init__(self, width=32):

        super().__init__()

        self.fc0 = nn.Linear(4, width)

        self.block1 = KANFNOBlock(width)
        self.block2 = KANFNOBlock(width)
        self.block3 = KANFNOBlock(width)

        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, 3)

        self.activation = nn.GELU()

    def forward(self, x):

        b, n, d = x.shape

        x = self.fc0(x)

        x = x.permute(0, 2, 1).unsqueeze(-1)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = x.squeeze(-1).permute(0, 2, 1)

        x = self.activation(self.fc1(x))

        return self.fc2(x)
