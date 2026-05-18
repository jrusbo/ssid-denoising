import torch.nn as nn


class LoNPE(nn.Module):
    """
    Locally Noise Prior Estimation (LoNPE) Module.
    Official implementation as per Condformer (IJCV 2025).
    Estimates a 2-channel dense spatial noise prior map (shot noise, read noise) from the noisy input.
    """

    def __init__(self, in_channels=3, mid_channels=32, out_channels=2):
        super(LoNPE, self).__init__()
        # Official architecture: 3x3 Conv -> ReLU -> 3x3 Conv -> ReLU -> 3x3 Conv -> Sigmoid
        self.estimation = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.estimation(x)
