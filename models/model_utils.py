import torch
import torch.nn as nn
from typing import Optional


def from_homogeneous(points, eps: float = 1e-8):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / (points[..., -1:] + eps)

def make_grid(
    w: float,
    h: float,
    step_x: float = 1.0,
    step_y: float = 1.0,
    orig_x: float = 0,
    orig_y: float = 0,
    y_up: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    x, y = torch.meshgrid(
        [
            torch.arange(orig_x, w + orig_x, step_x, device=device),
            torch.arange(orig_y, h + orig_y, step_y, device=device),
        ],
        indexing="xy",
    )
    if y_up:
        y = y.flip(-2)
    grid = torch.stack((x, y), -1)
    return grid

def checkpointed(cls, do=True):
    """Adapted from the DISK implementation of Michał Tyszkiewicz."""
    # 确保输入的类型为torch.nn.Module
    assert issubclass(cls, torch.nn.Module)

    class Checkpointed(cls):
        def forward(self, *args, **kwargs):
            super_fwd = super(Checkpointed, self).forward
            if any((torch.is_tensor(a) and a.requires_grad) for a in args):
                return torch.utils.checkpoint.checkpoint(super_fwd, *args, **kwargs)
            else:
                return super_fwd(*args, **kwargs)

    return Checkpointed if do else cls


class FcBlock(nn.Module):
    def __init__(self, conf, dropout=0.30):
        super(FcBlock, self).__init__()
        self.in_dim = conf.output_dim
        self.mid_dim = conf.mid_dim
        self.out_dim = 3

        # fc layers
        self.fcs = nn.Sequential(
            nn.Linear(self.in_dim, self.mid_dim),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(dropout),
            nn.Linear(self.mid_dim, int(self.mid_dim / 2)),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(dropout),
            nn.Linear(int(self.mid_dim / 2), 128),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        out = self.fcs(x)
        return out


class ECAttention(nn.Module):
    def __init__(self, k_size=3):
        super(ECAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)  
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


def conv(in_planes, out_planes, batchNorm, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )

class FeatureFusion(nn.Module):
    def __init__(self, conf):
        super(FeatureFusion, self).__init__()
        input_dim = conf.input_dim
        mid_dim = conf.mid_dim
        output_dim = conf.output_dim
        self.conv1 = conv(input_dim, mid_dim, batchNorm=True, kernel_size=3, stride=2, dropout=0.2)
        self.conv1_1 = conv(mid_dim, mid_dim, batchNorm=True, kernel_size=3, stride=1, dropout=0.2)
        self.conv2 = conv(mid_dim, mid_dim, batchNorm=True, kernel_size=3, stride=2, dropout=0.2)
        self.conv2_1 = conv(mid_dim, mid_dim, batchNorm=True, kernel_size=3, stride=1, dropout=0.2)
        self.conv3 = conv(mid_dim, output_dim, batchNorm=True, kernel_size=3, stride=2, dropout=0.5)
    
    def forward(self, fea):
        out_conv1 = self.conv1_1(self.conv1(fea))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3(out_conv2)

        return out_conv3