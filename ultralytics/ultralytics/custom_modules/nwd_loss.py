import torch
import torch.nn as nn


class ShapeNWDLoss(nn.Module):
    def __init__(self, C=4.0):
        super().__init__()
        self.C = C

    def forward(self, pred, target):
        px, py, pw, ph = pred.unbind(-1)
        tx, ty, tw, th = target.unbind(-1)

        dx2 = (px - tx) ** 2
        dy2 = (py - ty) ** 2
        B = (pw - tw) ** 2 + (ph - th) ** 2
        hh = ph * th
        ww = pw * tw

        D = torch.sqrt(torch.clamp(hh * dx2 + ww * dy2 + B, min=1e-6))  # Clamped for stability
        loss = 1.0 - torch.exp(-D / self.C)
        return loss.mean()


# Unit Test
if __name__ == "__main__":
    pred = torch.tensor([[50, 50, 20, 10]], dtype=torch.float32)
    target = torch.tensor([[52, 48, 21, 11]], dtype=torch.float32)
    loss = ShapeNWDLoss()(pred, target)
    print("ShapeNWDLoss:", loss.item())  # Should output a scalar value