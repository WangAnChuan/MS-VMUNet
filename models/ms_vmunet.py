import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

# ==============================================================================
# 1. MultiScale Convolution Block (MS-Conv)
# ==============================================================================
class MultiScaleConv(nn.Module):
    """
    Multi-Scale Depthwise Convolution to inject local geometric inductive bias.
    Branch 1: 3x3 standard depthwise (r=1)
    Branch 2: 3x3 dilated depthwise (r=2)
    Branch 3: 3x3 dilated depthwise (r=3)
    Branch 4: 1x1 pointwise conv
    """
    def __init__(self, dim):
        super().__init__()
        sub_dim = dim // 4
        self.sub_dim = sub_dim
        self.b1 = nn.Conv2d(sub_dim, sub_dim, 3, padding=1, groups=sub_dim)
        self.b2 = nn.Conv2d(sub_dim, sub_dim, 3, padding=2, dilation=2, groups=sub_dim)
        self.b3 = nn.Conv2d(sub_dim, sub_dim, 3, padding=3, dilation=3, groups=sub_dim)
        self.b4 = nn.Conv2d(sub_dim, sub_dim, 1)
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

    def forward(self, x):
        s = self.sub_dim
        b1 = self.b1(x[:, 0:s])
        b2 = self.b2(x[:, s:2*s])
        b3 = self.b3(x[:, 2*s:3*s])
        b4 = self.b4(x[:, 3*s:])
        return self.fusion(torch.cat([b1, b2, b3, b4], dim=1)) + x

# ==============================================================================
# 2. Configurable Mamba Block: 4-Way vs 2-Way, MS-Conv vs Single-Conv
# ==============================================================================
class FlexibleScanMambaBlock(nn.Module):
    """
    Configurable Visual State Space Block supporting:
    - 4-Way 2D Cross-Scan (horizontal fwd/bwd + vertical fwd/bwd) vs 2-Way (horizontal only)
    - MultiScale Depthwise Conv vs Single 3x3 Depthwise Conv
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, use_4way=True, use_msconv=True):
        super().__init__()
        self.dim = dim
        self.use_4way = use_4way
        self.use_msconv = use_msconv

        if use_msconv:
            self.pre_conv = MultiScaleConv(dim)
        else:
            self.pre_conv = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU()
            )

        self.norm = nn.LayerNorm(dim)

        self.mamba_h_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_h_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

        if use_4way:
            self.mamba_v_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            self.mamba_v_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            self.proj = nn.Sequential(
                nn.Linear(dim * 4, dim),
                nn.LayerNorm(dim)
            )
        else:
            self.mamba_v_fwd = None
            self.mamba_v_bwd = None
            self.proj = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim)
            )

    def forward(self, x):
        x = self.pre_conv(x)
        B, C, H, W = x.shape
        skip = x

        # 1. Horizontal Trajectories (Row-major)
        x_h = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        x_h_norm = self.norm(x_h)
        out_h_fwd = self.mamba_h_fwd(x_h_norm)
        out_h_bwd = torch.flip(self.mamba_h_bwd(torch.flip(x_h_norm, dims=[1])), dims=[1])

        if self.use_4way:
            # 2. Vertical Trajectories (Column-major)
            x_v = x.permute(0, 3, 2, 1).contiguous().view(B, W * H, C)
            x_v_norm = self.norm(x_v)
            out_v_fwd = self.mamba_v_fwd(x_v_norm)
            out_v_bwd = torch.flip(self.mamba_v_bwd(torch.flip(x_v_norm, dims=[1])), dims=[1])

            out_v_fwd = out_v_fwd.view(B, W, H, C).permute(0, 2, 1, 3).contiguous().view(B, H * W, C)
            out_v_bwd = out_v_bwd.view(B, W, H, C).permute(0, 2, 1, 3).contiguous().view(B, H * W, C)
            fused = torch.cat([out_h_fwd, out_h_bwd, out_v_fwd, out_v_bwd], dim=-1)
        else:
            fused = torch.cat([out_h_fwd, out_h_bwd], dim=-1)

        out = self.proj(fused).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return skip + out

# ==============================================================================
# 3. Modular Skip Connections: HFD vs Plain vs Additive RAA vs Multiplicative Gate
# ==============================================================================
class ModularSkipConnection(nn.Module):
    """
    Configurable Skip Connection Module supporting B5 mechanism comparisons:
    - 'hfd': High-Frequency Difference Edge Residual Skip (Proposed)
    - 'plain': Direct skip connection (Identity Concat)
    - 'additive_raa': 5x5 DWConv Additive Smoothing (Old RAA)
    - 'multiplicative_gate': Attention/Sigmoid Gated Skip (Old BG-Skip)
    """
    def __init__(self, dim, skip_type="hfd"):
        super().__init__()
        self.skip_type = skip_type.lower()
        if self.skip_type == "hfd":
            self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
            self.edge_branch = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1, bias=False),
                nn.BatchNorm2d(dim)
            )
            self.alpha = nn.Parameter(torch.tensor(0.1))
        elif self.skip_type == "additive_raa":
            self.smooth_branch = nn.Sequential(
                nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1, bias=False),
                nn.BatchNorm2d(dim)
            )
            self.alpha = nn.Parameter(torch.tensor(0.1))
        elif self.skip_type == "multiplicative_gate":
            self.gate_branch = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, 1, bias=False),
                nn.BatchNorm2d(dim),
                nn.Sigmoid()
            )
        elif self.skip_type == "plain":
            pass
        else:
            raise ValueError(f"Unknown skip_type: {skip_type}")

    def forward(self, skip):
        if self.skip_type == "hfd":
            high_freq = skip - self.pool(skip)
            return skip + self.alpha * self.edge_branch(high_freq)
        elif self.skip_type == "additive_raa":
            return skip + self.alpha * self.smooth_branch(skip)
        elif self.skip_type == "multiplicative_gate":
            return skip * self.gate_branch(skip)
        else: # plain
            return skip

# Backward compatibility alias
HighFrequencyDifferenceSkip = lambda dim, use_hfd=True: ModularSkipConnection(dim, skip_type="hfd" if use_hfd else "plain")

# ==============================================================================
# 4. Next-Gen Modular MS-VMUNet Model
# ==============================================================================
class MSVMUNetNextGen(nn.Module):
    """
    Next-Generation Modular MS-VMUNet (MS-VMUNet-Lite v3)
    Supports full combinatorial ablations (B1, B2, B5, B6):
    - use_4way: bool (4-Way 2D Cross-Scan vs 2-Way horizontal scan)
    - use_msconv: bool (Multi-Scale Depthwise Conv vs single 3x3 Conv)
    - skip_type: str ('hfd', 'plain', 'additive_raa', 'multiplicative_gate')
    - dims: channel width schedule (default: [32, 64, 128, 256] -> 1.85M parameters)
    """
    def __init__(self, in_channels=3, num_classes=1, dims=[32, 64, 128, 256],
                 use_4way=True, use_msconv=True, use_hfd=True, skip_type=None):
        super().__init__()
        self.dims = dims
        self.use_4way = use_4way
        self.use_msconv = use_msconv

        if skip_type is not None:
            self.skip_type = skip_type
        else:
            self.skip_type = "hfd" if use_hfd else "plain"

        # Stem: 256x256
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
            nn.Conv2d(dims[0], dims[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU()
        )

        # Stage 1: 256x256
        if use_msconv:
            self.enc1 = nn.Sequential(
                MultiScaleConv(dims[0]),
                nn.Conv2d(dims[0], dims[0], 3, padding=1, bias=False),
                nn.BatchNorm2d(dims[0]),
                nn.GELU()
            )
            self.dec1_post = MultiScaleConv(dims[0])
        else:
            self.enc1 = nn.Sequential(
                nn.Conv2d(dims[0], dims[0], 3, padding=1, bias=False),
                nn.BatchNorm2d(dims[0]),
                nn.GELU()
            )
            self.dec1_post = nn.Sequential(
                nn.Conv2d(dims[0], dims[0], 3, padding=1, bias=False),
                nn.BatchNorm2d(dims[0]),
                nn.GELU()
            )

        self.down1 = nn.Conv2d(dims[0], dims[1], 2, stride=2)

        # Stage 2: 128x128
        self.enc2 = nn.Sequential(
            FlexibleScanMambaBlock(dims[1], use_4way=use_4way, use_msconv=use_msconv),
            nn.Conv2d(dims[1], dims[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(dims[1]),
            nn.GELU()
        )
        self.down2 = nn.Conv2d(dims[1], dims[2], 2, stride=2)

        # Stage 3: 64x64
        self.enc3 = nn.Sequential(
            FlexibleScanMambaBlock(dims[2], use_4way=use_4way, use_msconv=use_msconv),
            nn.Conv2d(dims[2], dims[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(dims[2]),
            nn.GELU()
        )
        self.down3 = nn.Conv2d(dims[2], dims[3], 2, stride=2)

        # Bottleneck: 32x32 (Dual Mamba)
        self.bottleneck = nn.Sequential(
            FlexibleScanMambaBlock(dims[3], use_4way=use_4way, use_msconv=use_msconv),
            FlexibleScanMambaBlock(dims[3], use_4way=use_4way, use_msconv=use_msconv)
        )

        # Skip Connections
        self.skip3 = ModularSkipConnection(dims[2], skip_type=self.skip_type)
        self.skip2 = ModularSkipConnection(dims[1], skip_type=self.skip_type)
        self.skip1 = ModularSkipConnection(dims[0], skip_type=self.skip_type)

        # Decoders
        self.up3 = nn.ConvTranspose2d(dims[3], dims[2], 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(dims[2] * 2, dims[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(dims[2]),
            nn.GELU(),
            FlexibleScanMambaBlock(dims[2], use_4way=use_4way, use_msconv=use_msconv)
        )

        self.up2 = nn.ConvTranspose2d(dims[2], dims[1], 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(dims[1] * 2, dims[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(dims[1]),
            nn.GELU(),
            FlexibleScanMambaBlock(dims[1], use_4way=use_4way, use_msconv=use_msconv)
        )

        self.up1 = nn.ConvTranspose2d(dims[1], dims[0], 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(dims[0] * 2, dims[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
            self.dec1_post
        )

        self.final_head = nn.Conv2d(dims[0], num_classes, 1)

    def forward(self, x):
        x = self.stem(x)
        e1 = self.enc1(x)
        x = self.down1(e1)

        e2 = self.enc2(x)
        x = self.down2(e2)

        e3 = self.enc3(x)
        x = self.down3(e3)

        b = self.bottleneck(x)

        s3 = self.skip3(e3)
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))

        s2 = self.skip2(e2)
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))

        s1 = self.skip1(e1)
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))

        return self.final_head(d1)

def get_nextgen_model(ablation_type="full"):
    """
    Factory for Next-Gen models:
    - 'full': 4-Way + MS-Conv + HFD-Skip
    - 'no_hfd': 4-Way + MS-Conv + Plain Skip
    - 'no_msconv': 4-Way + Single-Conv + HFD-Skip
    - 'no_4way': 2-Way + MS-Conv + HFD-Skip
    - 'base': 2-Way + Single-Conv + Plain Skip
    - 'skip_additive_raa': 4-Way + MS-Conv + Additive RAA
    - 'skip_multiplicative_gate': 4-Way + MS-Conv + Multiplicative Gate
    - 'vmunet_slim': 4-Way + Single-Conv + Plain Skip (Param-aligned baseline)
    """
    t = ablation_type.lower()
    if t == "full":
        return MSVMUNetNextGen(use_4way=True, use_msconv=True, skip_type="hfd")
    elif t in ["no_hfd", "plain"]:
        return MSVMUNetNextGen(use_4way=True, use_msconv=True, skip_type="plain")
    elif t == "no_msconv":
        return MSVMUNetNextGen(use_4way=True, use_msconv=False, skip_type="hfd")
    elif t == "no_4way":
        return MSVMUNetNextGen(use_4way=False, use_msconv=True, skip_type="hfd")
    elif t == "base":
        return MSVMUNetNextGen(use_4way=False, use_msconv=False, skip_type="plain")
    elif t == "skip_additive_raa":
        return MSVMUNetNextGen(use_4way=True, use_msconv=True, skip_type="additive_raa")
    elif t == "skip_multiplicative_gate":
        return MSVMUNetNextGen(use_4way=True, use_msconv=True, skip_type="multiplicative_gate")
    elif t == "vmunet_slim":
        return MSVMUNetNextGen(use_4way=True, use_msconv=False, skip_type="plain")
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")

# Alias for standard naming
MSVMUNet = MSVMUNetNextGen

