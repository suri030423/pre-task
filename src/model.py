import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    [Conv(3x3) → BN → ReLU] × 2 블록
    U-Net encoder/decoder에서 공통으로 사용하는 기본 convolution 블록
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Down(nn.Module):
    """
    Down-sampling 블록
    MaxPool로 해상도를 1/2로 줄인 뒤, DoubleConv로 채널 수를 확장
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Up(nn.Module):
    """
    Up-sampling 블록
    ConvTranspose2d로 해상도를 2배 키운 뒤
    encoder에서 넘어온 skip feature와 concat하여 DoubleConv 수행
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        in_channels: upsample된 feature + skip feature를 concat한 채널 수
        out_channels: upsample 이후에 출력할 채널 수
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        # up 후 skip과 concat -> 채널 수 in_channels (= out_channels + skip_channels)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 16,
    ):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = Down(base_channels, base_channels * 2)
        self.enc3 = Down(base_channels * 2, base_channels * 4)

        self.bridge = Down(base_channels * 4, base_channels * 8)

        self.dec3 = Up(base_channels * 8, base_channels * 4)
        self.dec2 = Up(base_channels * 4, base_channels * 2)
        self.dec1 = Up(base_channels * 2, base_channels)

        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.enc1(x)
        c2 = self.enc2(c1)
        c3 = self.enc3(c2)

        b = self.bridge(c3)

        d3 = self.dec3(b, c3)
        d2 = self.dec2(d3, c2)
        d1 = self.dec1(d2, c1)

        return self.head(d1)
