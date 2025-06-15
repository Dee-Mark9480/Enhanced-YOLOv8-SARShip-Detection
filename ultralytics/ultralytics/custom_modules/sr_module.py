import torch
import torch.nn as nn

class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.branch_1x1_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.branch_avg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.branch_1x1(x) + self.branch_1x1_3x3(x) + self.branch_avg(x))

    def reparameterize(self):
        # Fuse branches into a single 3x3 conv for inference
        kernel, bias = self._get_fused_params()
        fused_conv = nn.Conv2d(
            self.branch_1x1[0].weight.shape[1],
            self.branch_1x1[0].weight.shape[0],
            kernel_size=3,
            padding=1,
            bias=True
        )
        fused_conv.weight.data = kernel
        fused_conv.bias.data = bias
        return fused_conv

    def _get_fused_params(self):
        # Merge all branches into a 3x3 kernel (implementation omitted for brevity)
        pass

class SRBlock(nn.Module):
    def __init__(self, channels, groups=2):
        super().__init__()
        assert channels % 2 == 0, "Channels must be divisible by 2"
        self.repconv = RepConv(channels // 2, channels // 2)
        self.groups = groups

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat([x1, self.repconv(x2)], dim=1)
        return self._channel_shuffle(out, self.groups)

    @staticmethod
    def _channel_shuffle(x, groups):
        B, C, H, W = x.shape
        x = x.view(B, groups, C // groups, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.view(B, C, H, W)

# Unit Test
if __name__ == "__main__":
    x = torch.randn(1, 64, 64, 64)
    model = SRBlock(64)
    y = model(x)
    print("SRBlock output shape:", y.shape)  # Should be (1, 64, 64, 64)