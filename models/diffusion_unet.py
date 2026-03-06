import torch
import torch.nn as nn


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GELU()
        )

    def forward(self, x):

        return self.net(x)


class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.down1 = DoubleConv(3, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv1 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 3, 1)

    def forward(self, x):

        d1 = self.down1(x)

        d2 = self.down2(self.pool(d1))

        d3 = self.down3(self.pool(d2))

        u1 = self.up1(d3)

        u1 = torch.cat([u1, d2], dim=1)

        u1 = self.conv1(u1)

        u2 = self.up2(u1)

        u2 = torch.cat([u2, d1], dim=1)

        u2 = self.conv2(u2)

        return self.out(u2)
