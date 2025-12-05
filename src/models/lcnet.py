import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicThresholdConv(nn.Module):
    """
    Section 3.1: Dynamic Threshold Convolution (DTConv)
    Changes from original:
    1. Added Linear Transformation after split [cite: 257]
    2. Changed main dynamic kernels to Standard Conv 3x3 (Dense) to match parameter counts [cite: 186, 433]
    3. Implemented 'num_experts' (N) kernels, selecting Top-k [cite: 271]
    """
    def __init__(self, in_channels: int, k: int = 2, num_experts: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.k = k
        self.num_experts = num_experts # N in Eq (1)

        # 1. Split channels logic
        div = in_channels // 3
        rem = in_channels % 3
        self.split_channels = [div + 1 if i < rem else div for i in range(3)]

        # --- Fix 1: Linear Transformation Matrix (Eq 6) [cite: 257] ---
        # "A linear transformation matrix Wi is then defined..."
        self.proj_split = nn.ModuleList([
            nn.Conv2d(c, c, 1) for c in self.split_channels
        ])

        # MSMod1: Multi-scale convolutions for weight generation [cite: 262]
        self.msmod1_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.split_channels[0], self.split_channels[0], 3, padding=1, groups=self.split_channels[0]),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(self.split_channels[0]) # BN after activation as per Eq 8 order in paper implies BN(F_i)
            ),
            nn.Sequential(
                nn.Conv2d(self.split_channels[1], self.split_channels[1], 5, padding=2, groups=self.split_channels[1]),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(self.split_channels[1])
            ),
            nn.Sequential(
                nn.Conv2d(self.split_channels[2], self.split_channels[2], 7, padding=3, groups=self.split_channels[2]),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(self.split_channels[2])
            )
        ])

        # --- Fix 2: Dynamic Kernels (Standard Dense Convs) ---
        # Figure 2 shows "Conv 3x3 Kd". 
        # To reach ~75M params (LCNet-B), these must be Dense, not Depthwise.
        # We define N parallel experts.
        self.experts = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=1) # Dense Conv
            for _ in range(num_experts)
        ])

        # Dynamic Weight Head
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_weights = nn.Linear(in_channels, num_experts) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        
        # A. Split & Linear Transform 
        splits = torch.split(x, self.split_channels, dim=1)
        splits_proj = [proj(s) for proj, s in zip(self.proj_split, splits)]
        
        # B. MSMod1 (Multi-scale Feature Extraction) [cite: 263]
        feats = [m(s) for m, s in zip(self.msmod1_branches, splits_proj)]
        x_multi = torch.cat(feats, dim=1) # [cite: 268]
        
        # C. Generate Weights
        global_feat = self.global_avg_pool(x_multi).view(b, -1)
        weights = self.fc_weights(global_feat)
        weights = F.softmax(weights, dim=1) # [cite: 273]

        # D. Top-k Selection & Re-normalization [cite: 270]
        if self.k < self.num_experts:
            topk_vals, topk_indices = torch.topk(weights, self.k, dim=1)
            mask = torch.zeros_like(weights)
            mask.scatter_(1, topk_indices, topk_vals)
            weights = mask
            # Stability Fix: Re-normalize
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # E. Apply Dynamic Kernels [cite: 274]
        # Sum(alpha_i * K_i(x))
        out = torch.zeros_like(x)  # Initialize as zero tensor with same shape as x
        for i in range(self.num_experts):
            w_i = weights[:, i].view(b, 1, 1, 1)
            # Only compute if weight is significant (optimization) or compute all for simplicity in training
            out += w_i * self.experts[i](x)
              
        return out


class MultipathDynamicAttention(nn.Module):
    """
    Section 3.2: Multi-path Dynamic Attention Mechanism (MDAM)
    Changes:
    1. Added flexible embed_dim logic for Base variant 
    """
    def __init__(self, in_channels: int, embed_dim: int = 64, k_ratio: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.k_ratio = k_ratio 

        # MSMod2 [cite: 326]
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels)

        # Projections [cite: 336]
        # Input is 3*C (concatenated features)
        self.q_proj = nn.Linear(3 * in_channels, embed_dim)
        self.k_proj = nn.Linear(3 * in_channels, embed_dim)
        self.v_proj = nn.Linear(3 * in_channels, in_channels)

        self.out_proj = nn.Linear(in_channels, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        N = h * w

        # 1. Multi-scale Features
        f1 = self.conv1x1(x)
        f2 = self.conv3x3(x)
        f3 = self.conv5x5(x)
        f_multi = torch.cat([f1, f2, f3], dim=1) 
        
        f_flat = f_multi.flatten(2).transpose(1, 2) # (B, N, 3C)

        # 2. QKV
        Q = self.q_proj(f_flat)
        K = self.k_proj(f_flat)
        V = self.v_proj(f_flat)

        # 3. Attention Map [cite: 344]
        attn = (Q @ K.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        # 4. Top-k Selection [cite: 351]
        key_importance = attn.sum(dim=1) 
        k_val = int(N * self.k_ratio)
        k_val = max(1, min(k_val, N))
        _, topk_indices = torch.topk(key_importance, k_val, dim=1)
        
        # 5. Attention Correction [cite: 360]
        mask = torch.zeros_like(key_importance)
        mask.scatter_(1, topk_indices, 1.0) 
        mask = mask.unsqueeze(1)

        attn_corrected = attn * mask
        # Stability Fix: Re-normalize
        attn_corrected = attn_corrected / (attn_corrected.sum(dim=-1, keepdim=True) + 1e-8)

        out = attn_corrected @ V
        out = self.out_proj(out)
        out = out.transpose(1, 2).view(b, c, h, w)
        return out


class LCNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, embed_dim: int = 64):
        super().__init__()
        
        # Eq 5: X1 = BN(Conv3x3(X)) [cite: 176]
        self.conv_start = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Eq 5: DWConv(X1) - Shared input to branches [cite: 176]
        # Paper explicitly says DWConv here
        self.dw_shared = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Dual Path
        self.dtconv = DynamicThresholdConv(in_channels)
        self.mdam = MultipathDynamicAttention(in_channels, embed_dim=embed_dim)

        # Output Trans: Conv1x1 -> DWConv [cite: 176]
        self.output_trans = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x1 = self.conv_start(x)
        x_shared = self.dw_shared(x1)

        out_dt = self.dtconv(x_shared)
        out_mdam = self.mdam(x_shared)

        # Star Operation [cite: 397]
        x2 = out_dt * out_mdam
        x2 = torch.clamp(x2, min=-10.0, max=10.0) # Stability

        x3 = self.output_trans(x2)
        return x3 + residual


class LCNet(nn.Module):
    def __init__(
        self, num_classes: int = 10, variant: Literal["tiny", "small", "base"] = "base"
    ):
        super().__init__()
        
        # --- Fix 3: Variant Configs ---
        # "Only vary the embed width and the depth..." 
        # Base needs larger embed_dim to reach parameter count targets.
        if variant == "tiny":
            stem_out = 24
            embed_dim = 64
            stage_configs = [
                (24, 2, 1), (48, 2, 2), (96, 8, 2), (192, 2, 2)
            ]
        elif variant == "small":
            stem_out = 32
            embed_dim = 64  # Increased
            stage_configs = [
                (32, 2, 1), (64, 4, 2), (128, 12, 2), (256, 2, 2)
            ]
        elif variant == "base":
            stem_out = 32
            embed_dim = 64 # Increased for Base
            stage_configs = [
                (32, 4, 1),
                (64, 6, 2),
                (128, 16, 2),
                (256, 4, 2),
            ]
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_out, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_out, stem_out, 3, stride=2, padding=1, groups=stem_out, bias=False),
            nn.BatchNorm2d(stem_out),
            nn.ReLU(inplace=True),
        )

        self.stages = nn.ModuleList()
        in_ch = stem_out
        
        for out_ch, num_layers, stage_stride in stage_configs:
            layers = []
            for i in range(num_layers):
                stride = stage_stride if i == 0 else 1
                layers.append(LCNetBlock(in_ch, out_ch, stride=stride, embed_dim=embed_dim))
                in_ch = out_ch
            self.stages.append(nn.Sequential(*layers))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.gap(x).flatten(1)
        x = self.fc(x)
        return x