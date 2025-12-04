import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicThresholdConv(nn.Module):
    """
    Section 3.1: Dynamic Threshold Convolution (DTConv) [cite: 255]
    Process:
    1. Split Input -> Linear Trans -> MSMod1 (3x3, 5x5, 7x7) -> Concat -> Weights [cite: 257-270]
    2. Top-k Selection of weights [cite: 270]
    3. Weighted Sum of Kernels (simulated by weighting outputs) [cite: 274]
    """
    def __init__(self, in_channels: int, k: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.k = k

        # 1. Split channels into 3 parts for weight generation 
        div = in_channels // 3
        rem = in_channels % 3
        self.split_channels = [div + 1 if i < rem else div for i in range(3)]

        # MSMod1: Multi-scale convolutions for weight generating features [cite: 262]
        # Group 1: 3x3
        self.conv3x3_split = nn.Sequential(
            nn.Conv2d(self.split_channels[0], self.split_channels[0], 3, padding=1, groups=self.split_channels[0]),
            nn.BatchNorm2d(self.split_channels[0]),
            nn.ReLU(inplace=True),
        )
        # Group 2: 5x5
        self.conv5x5_split = nn.Sequential(
            nn.Conv2d(self.split_channels[1], self.split_channels[1], 5, padding=2, groups=self.split_channels[1]),
            nn.BatchNorm2d(self.split_channels[1]),
            nn.ReLU(inplace=True),
        )
        # Group 3: 7x7
        self.conv7x7_split = nn.Sequential(
            nn.Conv2d(self.split_channels[2], self.split_channels[2], 7, padding=3, groups=self.split_channels[2]),
            nn.BatchNorm2d(self.split_channels[2]),
            nn.ReLU(inplace=True),
        )

        # Main convolutions to be weighted (Simulating dynamic kernels H_i)
        # Since summing kernels of different sizes needs padding, we sum outputs of parallel convs.
        self.main_conv3x3 = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.main_conv5x5 = nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels)
        self.main_conv7x7 = nn.Conv2d(in_channels, in_channels, 7, padding=3, groups=in_channels)

        # Dynamic Weight Generation Head
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_weights = nn.Linear(in_channels, 3) # 3 scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        
        # --- A. Generate Dynamic Weights (based on Split + MSMod1) ---
        x1, x2, x3 = torch.split(x, self.split_channels, dim=1) # 
        
        f1 = self.conv3x3_split(x1)
        f2 = self.conv5x5_split(x2)
        f3 = self.conv7x7_split(x3)
        
        x_multi = torch.cat([f1, f2, f3], dim=1) # [cite: 268]
        
        # Flatten -> Softmax -> Weights [cite: 270]
        global_feat = self.global_avg_pool(x_multi).view(b, -1)
        weights = self.fc_weights(global_feat) # (B, 3)
        weights = F.softmax(weights, dim=1)

        # --- B. Top-k Selection ---
        # "Select top k weights based on predefined threshold" [cite: 270]
        if self.k < 3:
            topk_vals, topk_indices = torch.topk(weights, self.k, dim=1)
            mask = torch.zeros_like(weights)
            mask.scatter_(1, topk_indices, topk_vals) # Keep Top-k values, others 0
            weights = mask
            # Re-normalize to prevent NaN
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # --- C. Apply Dynamic Weights ---
        # G = X * Sum(W_i * H_i) [cite: 274]
        w1 = weights[:, 0].view(b, 1, 1, 1)
        w2 = weights[:, 1].view(b, 1, 1, 1)
        w3 = weights[:, 2].view(b, 1, 1, 1)

        # Convolve input with 3 scales and weight them
        out = (self.main_conv3x3(x) * w1) + \
              (self.main_conv5x5(x) * w2) + \
              (self.main_conv7x7(x) * w3)
              
        return out


class MultipathDynamicAttention(nn.Module):
    """
    Section 3.2: Multi-path Dynamic Attention Mechanism (MDAM) [cite: 323]
    Process:
    1. MSMod2 (1x1, 3x3, 5x5) -> Concat -> QKV [cite: 326-338]
    2. Attention -> Sort -> Select Top-k Tokens [cite: 351]
    3. Attention Correction (Masking) [cite: 357]
    """
    def __init__(self, in_channels: int, embed_dim: int = 64, k_ratio: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.k_ratio = k_ratio # Represents "k" tokens as a ratio of total pixels

        # MSMod2: Multi-scale feature extraction [cite: 326]
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels)

        # Q, K, V Projections (Input is 3 * C from concat) [cite: 336]
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
        f_multi = torch.cat([f1, f2, f3], dim=1) # (B, 3C, H, W) [cite: 332]
        
        f_flat = f_multi.flatten(2).transpose(1, 2) # (B, N, 3C)

        # 2. Generate Q, K, V
        Q = self.q_proj(f_flat)
        K = self.k_proj(f_flat)
        V = self.v_proj(f_flat)

        # 3. Dynamic Modeling (Attention Map)
        # a_t = Softmax(QK^T / sqrt(d)) [cite: 344]
        attn = (Q @ K.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn = F.softmax(attn, dim=-1) # (B, N, N)

        # 4. Token Selection (Top-k) [cite: 351]
        # Calculate total attention for each key token
        key_importance = attn.sum(dim=1) # Sum over queries (B, N)
        
        k_val = int(N * self.k_ratio)
        k_val = max(1, min(k_val, N))

        # Sort and select Top-k indices
        _, topk_indices = torch.topk(key_importance, k_val, dim=1)
        
        # 5. Attention Correction 
        # Create mask: 1 for Top-k (alpha=1), 0 for others
        mask = torch.zeros_like(key_importance)
        mask.scatter_(1, topk_indices, 1.0) 
        mask = mask.unsqueeze(1) # Broadcast (B, 1, N)

        attn_corrected = attn * mask
        # Re-normalize attention to prevent NaN
        attn_corrected = attn_corrected / (attn_corrected.sum(dim=-1, keepdim=True) + 1e-8) 

        # Output Generation
        out = attn_corrected @ V
        out = self.out_proj(out)
        out = out.transpose(1, 2).view(b, c, h, w)
        
        return out


class LCNetBlock(nn.Module):
    """
    Structure based on Eq (5) and Fig 2 [cite: 176, 247]
    Order:
    1. Conv 3x3 (Input processing)
    2. DWConv (Shared extraction)
    3. Dual Path (DTConv // MDAM)
    4. Star Fusion
    5. Conv 1x1
    6. DWConv (Output processing)
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        # Eq 5: X1 = BN(Conv3x3(X)) 
        # This layer also handles the stride if needed
        self.conv_start = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Eq 5: "... fed into DTConv/MDAM(DWConv(X1))" 
        self.dw_shared = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Dual Path Branches
        self.dtconv = DynamicThresholdConv(in_channels)
        self.mdam = MultipathDynamicAttention(in_channels)

        # Eq 5: X3 = DWConv(Conv1x1(X2)) 
        self.output_trans = nn.Sequential(
            # Conv 1x1
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # DWConv
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels)
            # No ReLU at end of block usually to allow full residual range
        )

        # Residual Shortcut
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        # 1. Conv 3x3 & DWConv
        x1 = self.conv_start(x)
        x_shared = self.dw_shared(x1)

        # 2. Dual Path
        out_dt = self.dtconv(x_shared)
        out_mdam = self.mdam(x_shared)

        # 3. Star Operation [cite: 397]
        # "Element-wise multiplication in high-dimensional space"
        x2 = out_dt * out_mdam
        
        # Clamp to prevent extreme values that cause NaN in BatchNorm
        x2 = torch.clamp(x2, min=-10.0, max=10.0)

        # 4. Output Transformation
        x3 = self.output_trans(x2)

        # 5. Residual Add
        return x3 + residual


class LCNet(nn.Module):
    def __init__(self, num_classes: int = 10, variant: str = "tiny"):
        super().__init__()
        
        # Configuration based on Table 1 
        if variant == "tiny":
            stem_out = 24
            # Stage Config: (out_channels, num_blocks, stride_of_first_block)
            # Stride logic: Input 224 -> Stem(56) -> Stage1(56) -> Stage2(28) -> Stage3(14) -> Stage4(7)
            stage_configs = [
                (24, 2, 1),   # Stage 1: 56x56 -> 56x56
                (48, 2, 2),   # Stage 2: 56x56 -> 28x28 (Stride 2)
                (96, 8, 2),   # Stage 3: 28x28 -> 14x14 (Stride 2)
                (192, 2, 2),  # Stage 4: 14x14 -> 7x7 (Stride 2)
            ]
        elif variant == "small":
            stem_out = 32
            stage_configs = [
                (32, 2, 1),
                (64, 4, 2),
                (128, 12, 2),
                (256, 2, 2),
            ]
        elif variant == "base":
            stem_out = 32
            stage_configs = [
                (32, 4, 1),
                (64, 6, 2),
                (128, 16, 2),
                (256, 4, 2),
            ]
        
        # Stem: Input 224 -> Output 56 (Downsample /4) 
        # Standard approach: Two stride-2 convolutions
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
                layers.append(LCNetBlock(in_ch, out_ch, stride=stride))
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