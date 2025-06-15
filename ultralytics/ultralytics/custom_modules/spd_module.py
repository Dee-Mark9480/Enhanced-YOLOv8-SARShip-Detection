import torch
import torch.nn as nn

class SpaceToDepth(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.size()
        assert H % 2 == 0 and W % 2 == 0, "Input dimensions must be divisible by 2"
        x = x.view(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * 4, H // 2, W // 2)
        return x

class SPDModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.spd = SpaceToDepth()
        self.conv1x1 = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)  # Changed from ReLU to SiLU (YOLOv8 default)

    def forward(self, x):
        x = self.act(self.bn1(self.conv3x3(x)))
        x = self.spd(x)
        x = self.act(self.bn2(self.conv1x1(x)))
        return x

# Unit Test
if __name__ == "__main__":
    x = torch.randn(1, 32, 64, 64)
    model = SPDModule(32, 64)
    y = model(x)
    print("SPDModule output shape:", y.shape)  # Should be (1, 64, 32, 32)