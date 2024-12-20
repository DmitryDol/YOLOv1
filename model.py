"""
Implementation of YOLOv1 architecture from original paper
with slight modification: added BatchNorm
"""

import torch
import torch.nn as nn


architecture_config = [
    # Tuple: (kernel_size, num_filters, stride, padding)
    (7, 64, 2, 3),
    # MaxPool2d(kernel_size=2, stride=2),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    # List: [(kernel_size, num_filters, stride, padding), repeat]
    # List: tuples and then last int represents number of repeats
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leakyrelu(self.batchnorm(self.conv(x)))


# from scratch
class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(YOLOv1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        # conv layers
        self.darknet = self._create_conv_layers(self.architecture)
        # fully connected layers
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture: list) -> nn.Sequential:
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],)
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size: int, num_boxes: int, num_classes: int) -> nn.Sequential:
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496), # In original paper it's 4096, but it`s quite a lot for me
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)), # (S, S, 30) where 30 = C + B * 5
        )
    
    
def test(S: int=7, B: int=2, C: int=20) -> torch.Size:
    model = YOLOv1(split_size=S, num_boxes=B, num_classes=C)
    x = torch.randn((2, 3, 448, 448))
    return model(x).shape

if __name__ == '__main__':
    print(test())