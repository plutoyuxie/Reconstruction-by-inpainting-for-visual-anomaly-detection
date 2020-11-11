import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import kornia

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


# Define Prewitt operator:
class Prewitt(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)
        Gx = torch.tensor([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]) / 3
        Gy = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]) / 3
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1).to(device)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x


# Define the gradient magnitude similarity map:
def GMS(Ii, Ir, edge_filter, median_filter, c=0.0026):
    x = torch.mean(Ii, dim=1, keepdim=True)
    y = torch.mean(Ir, dim=1, keepdim=True)
    g_I = edge_filter(median_filter(x))
    g_Ir = edge_filter(median_filter(y))
    g_map = (2 * g_I * g_Ir + c) / (g_I**2 + g_Ir**2 + c)
    return g_map


class MSGMS_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.GMS = partial(GMS, edge_filter=Prewitt(), median_filter=kornia.filters.MedianBlur((3, 3)))

    def GMS_loss(self, Ii, Ir):
        return torch.mean(1 - self.GMS(Ii, Ir))

    def forward(self, Ii, Ir):
        total_loss = self.GMS_loss(Ii, Ir)

        for _ in range(3):
            Ii = F.avg_pool2d(Ii, kernel_size=2, stride=2)
            Ir = F.avg_pool2d(Ir, kernel_size=2, stride=2)
            total_loss += self.GMS_loss(Ii, Ir)

        return total_loss / 4


class MSGMS_Score(nn.Module):
    def __init__(self):
        super().__init__()
        self.GMS = partial(GMS, edge_filter=Prewitt(), median_filter=kornia.filters.MedianBlur((3, 3)))
        self.median_filter = kornia.filters.MedianBlur((21, 21))

    def GMS_Score(self, Ii, Ir):
        return self.GMS(Ii, Ir)

    def forward(self, Ii, Ir):
        total_scores = self.GMS_Score(Ii, Ir)
        img_size = Ii.size(-1)
        total_scores = F.interpolate(total_scores, size=img_size, mode='bilinear', align_corners=False)
        for _ in range(3):
            Ii = F.avg_pool2d(Ii, kernel_size=2, stride=2)
            Ir = F.avg_pool2d(Ir, kernel_size=2, stride=2)
            score = self.GMS_Score(Ii, Ir)
            total_scores += F.interpolate(score, size=img_size, mode='bilinear', align_corners=False)

        return (1 - total_scores) / 4
