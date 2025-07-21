from typing import Tuple

import torch
import torch.nn as nn

from conv import Conv


class Bottleneck(nn.Module):
    """Standard bottleneck block - optimized version."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1,
                 k: Tuple[int, int] = (3, 3), e: float = 0.5):
        """Initialize Bottleneck with given channels, shortcut, groups, kernels, and expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through bottleneck block."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class GhostConv(nn.Module):
    """Ghost Convolution - optimized for efficiency."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initialize Ghost Convolution with reduced parameters."""
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward pass through Ghost Convolution."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)

    def forward_fuse(self, x):
        """Fused forward pass."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck - simplified for better efficiency."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initialize GhostBottleneck with optimized structure."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            GhostConv(c_, c_, 3, s, g=c_) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (nn.Sequential(
            nn.Conv2d(c1, c2, 3, s, 1, bias=False),
            nn.BatchNorm2d(c2)
        ) if s == 2 or c1 != c2 else nn.Identity())

    def forward(self, x):
        """Forward pass through Ghost Bottleneck."""
        return self.conv(x) + self.shortcut(x)


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions - base implementation."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True,
                 g: int = 1, e: float = 0.5):
        """Initialize CSP Bottleneck."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CSP bottleneck."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3Ghost(C3):
    """C3 module with GhostBottleneck - optimized for fewer parameters."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost with optimized structure."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # Use simpler bottleneck structure for Ghost version
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class C3k2(C3):
    """C3k2 module with configurable kernel sizes - optimized."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initialize C3k2 with kernel size options."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        if c3k:
            self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))
        else:
            self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False,
                 g: int = 1, e: float = 0.5):
        """Initialize CSP bottleneck with 2 convolutions."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class DFL(nn.Module):
    """Distribution Focal Loss (DFL) integral module."""

    def __init__(self, c1: int = 16):
        """Initialize DFL module."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DFL."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)