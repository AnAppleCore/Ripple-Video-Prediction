import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class ConvStep(nn.Module):
    def __init__(self, in_c, out_c, k, s, p):
        super(ConvStep, self).__init__()
        self.step = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, 
            stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.step(x)


class RippleHeight(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(RippleHeight, self).__init__()

        self.encoder = nn.Sequential(
            ConvStep(in_c=in_channels, out_c=64, k=3, s=1, p=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvStep(in_c=64, out_c=128, k=3, s=1, p=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvStep(in_c=128, out_c=256, k=3, s=1, p=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        y = self.encoder(x)
        y = self.decoder(y)
        y = torch.squeeze(y, 1)
        return y


def main():
    # test RippleHeight model
    x = torch.randn((4, 3, 256, 256), dtype=torch.float32) # (batch, c, h, w)
    model = RippleHeight()
    h = model(x)
    print(x.shape, h.shape)


if __name__ == '__main__':
    main()