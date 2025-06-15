import torch
import torch.nn as nn
import torch.fft


class MEW(nn.Module):
    def __init__(self, channels):
        super().__init__()
        assert channels % 4 == 0, "Channels must be divisible by 4"
        chunk = channels // 4
        self.weight_HW = nn.Parameter(torch.ones(1, chunk, 1, 1))
        self.weight_CW = nn.Parameter(torch.ones(1, chunk, 1, 1))
        self.weight_CH = nn.Parameter(torch.ones(1, chunk, 1, 1))
        self.dw_conv = nn.Conv2d(chunk, chunk, kernel_size=3, padding=1, groups=chunk)

    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        # Frequency-domain attention (H-W, C-W, C-H)
        x1 = torch.fft.ifft2(torch.fft.fft2(x1) * self.weight_HW).real
        x2 = torch.fft.ifft2(torch.fft.fft2(x2) * self.weight_CW).real
        x3 = torch.fft.ifft2(torch.fft.fft2(x3) * self.weight_CH).real

        # Spatial-domain attention (DW conv)
        x4 = self.dw_conv(x4)

        return torch.cat([x1, x2, x3, x4], dim=1) + x  # Residual


class SA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(1, channels, kernel_size=1),
            nn.Sigmoid()  # Added sigmoid for [0,1] scaling
        )

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        attn = self.conv(torch.cat([max_pool, avg_pool], dim=1))
        return x * attn  # Scaled output


class HybridAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mew = MEW(channels)
        self.sa = SA(channels)

    def forward(self, x):
        x = self.mew(x)
        return self.sa(x)


# Unit Test
if __name__ == "__main__":
    x = torch.randn(1, 64, 64, 64)
    model = HybridAttention(64)
    y = model(x)
    print("HybridAttention output shape:", y.shape)  # Should be (1, 64, 64, 64)