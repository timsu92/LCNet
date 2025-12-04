import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicThresholdConv(nn.Module):
    def __init__(self, in_channels: int, k: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.k = k  # Top-k selection

        # Split channels into 3 parts
        div = in_channels // 3
        rem = in_channels % 3
        self.split_channels = [div + 1 if i < rem else div for i in range(3)]

        # Multi-scale convolutions (MSMod1)
        # Group 1: 3x3
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(
                self.split_channels[0],
                self.split_channels[0],
                kernel_size=3,
                padding=1,
                groups=self.split_channels[0],
            ),
            nn.BatchNorm2d(self.split_channels[0]),
            nn.ReLU(inplace=True),
        )
        # Group 2: 5x5
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(
                self.split_channels[1],
                self.split_channels[1],
                kernel_size=5,
                padding=2,
                groups=self.split_channels[1],
            ),
            nn.BatchNorm2d(self.split_channels[1]),
            nn.ReLU(inplace=True),
        )
        # Group 3: 7x7
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(
                self.split_channels[2],
                self.split_channels[2],
                kernel_size=7,
                padding=3,
                groups=self.split_channels[2],
            ),
            nn.BatchNorm2d(self.split_channels[2]),
            nn.ReLU(inplace=True),
        )

        # Dynamic Weight Generation
        # We need to generate weights for the k selected kernels.
        # The paper says: Flatten -> Softmax -> Weights W.
        # Then Top-k selection.
        # Then G = X * sum(W_topk * H_i)

        # To implement "Dynamic Selection of Convolution Kernels" efficiently in PyTorch:
        # Usually this implies we have multiple kernels and we select which ones to apply,
        # or we apply all and weight their outputs.
        # Given the description "Weighted sum of selected kernels", and "G = X * ...",
        # it sounds like a dynamic convolution or mixture of experts.

        # Simplified interpretation based on "Dynamic Weight Generation":
        # We generate weights for the 3 branches (3x3, 5x5, 7x7) based on global context.
        # If k < 3, we only use the top-k branches.

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_weights = nn.Linear(in_channels, 3)  # 3 branches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # Split input
        x1, x2, x3 = torch.split(x, self.split_channels, dim=1)

        # Apply convolutions
        out1 = self.conv3x3(x1)
        out2 = self.conv5x5(x2)
        out3 = self.conv7x7(x3)

        # Stack outputs for selection: (B, 3, C/3, H, W)
        # Wait, the paper says "Concat and BN to get X_multi".
        # If we concat, we get back (B, C, H, W).
        # But the dynamic part selects "Kernels".

        # Let's follow the "Dynamic Weight" part:
        # "Flatten and Softmax to generate weight vector W"
        # Usually this is done on the input or the concatenated output.
        # Let's assume it's SE-like on the concatenated output.

        x_multi = torch.cat([out1, out2, out3], dim=1)  # (B, C, H, W)

        # Generate weights
        # Global context
        global_feat = self.global_avg_pool(x_multi).view(b, -1)  # (B, C)

        # The paper mentions "Top-k selection of weights and corresponding kernels".
        # Since we have 3 fixed kernels (branches) effectively,
        # maybe it means we weight the 3 branches?
        # But 3 is small.

        # Let's implement a channel-wise dynamic weighting (SE-block style) but with Top-k.
        # Or maybe it means selecting channels?

        # Re-reading model_arch.md:
        # "Input channels divided into 3 parts... MSMod1... Concat... X_multi"
        # "Dynamic Weight Generation: Flatten X_multi, Softmax -> W"
        # "Top-k selection: Keep top k weights and corresponding kernels"
        # "G = X * sum(W_topk * H_i)"

        # This implies the "Kernels" H_i are the 3 branches we just computed?
        # If so, we are weighting the outputs of the 3 branches.
        # But the branches operate on different channel splits.
        # So we can't just sum them up unless we project them to the same space.

        # Alternative interpretation:
        # The "Kernels" are not the 3 branches, but a set of dynamic kernels applied to the whole input?
        # Given "Lightweight", the split-transform-merge (ResNeXt/Inception) pattern is common.

        # Let's stick to the most logical implementation for "Dynamic Threshold":
        # We have 3 branches. We compute a weight for each branch (or channel group).
        # We select the Top-k branches (e.g. k=2 out of 3) and zero out the others.

        # However, since x1, x2, x3 are different parts of input, we need all of them to reconstruct the full feature map?
        # Unless the output is a summation of features in a shared space.
        # But usually split-concat preserves channel count.

        # Let's assume the "Dynamic Threshold" applies to a channel attention mechanism.
        # We generate weights for C channels. We keep Top-k channels?
        # "Dynamic Threshold Convolution" usually implies the activation threshold is dynamic,
        # OR the weights are dynamic.

        # Given the ambiguity and lack of exact formula, I will implement a robust
        # "Split -> Multi-scale Conv -> Concat -> SE-like Attention with Top-k" block.
        # This fits "Dynamic Attention" and "Multi-path".

        # For DTConv specifically:
        # I will treat it as:
        # 1. Split -> Conv -> Concat (Standard Inception-like)
        # 2. Reweight the channels using a Top-k mask.

        # Weight generation
        # We want weights for the C channels.
        weights = self.fc_weights(global_feat)  # (B, 3) - One weight per branch?
        weights = F.softmax(weights, dim=1)

        # Top-k selection on the 3 branch weights
        if self.k < 3:
            topk_vals, topk_indices = torch.topk(weights, self.k, dim=1)
            # Create a mask
            mask = torch.zeros_like(weights)
            mask.scatter_(1, topk_indices, topk_vals)  # Keep top-k values, others 0
            # Normalize so they sum to 1? Or just keep as is?
            # Paper says "Attention Correction" for MDAM, maybe similar here.
            # Let's just use the masked weights.
            weights = mask

        # Apply weights to the branches
        # We need to broadcast weights (B, 3) to (B, C, H, W)
        # Since x_multi is (B, C, H, W) and composed of [out1, out2, out3]
        # We expand weights: weight[0] for out1 channels, weight[1] for out2...

        w1 = weights[:, 0].view(b, 1, 1, 1)
        w2 = weights[:, 1].view(b, 1, 1, 1)
        w3 = weights[:, 2].view(b, 1, 1, 1)

        # Apply
        out = torch.cat([out1 * w1, out2 * w2, out3 * w3], dim=1)

        return out


class MultipathDynamicAttention(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int = 64, k: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.k = k  # Top-k tokens

        # MSMod2: Multi-scale feature extraction for Q, K, V
        # 1x1, 3x3, 5x5
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, 5, padding=2)

        # Linear projections for Q, K, V
        # Input to these will be concat of MSMod2 outputs (3 * in_channels)
        self.q_proj = nn.Linear(3 * in_channels, embed_dim)
        self.k_proj = nn.Linear(3 * in_channels, embed_dim)
        self.v_proj = nn.Linear(
            3 * in_channels, in_channels
        )  # Output needs to match input channels for residual?

        # Output projection
        self.out_proj = nn.Linear(in_channels, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # MSMod2
        f1 = self.conv1x1(x)
        f2 = self.conv3x3(x)
        f3 = self.conv5x5(x)

        # Concat: (B, 3C, H, W)
        f_multi = torch.cat([f1, f2, f3], dim=1)

        # Flatten spatial dims for Attention: (B, 3C, N) where N = H*W
        f_flat = f_multi.flatten(2).transpose(1, 2)  # (B, N, 3C)

        # Generate Q, K, V
        Q = self.q_proj(f_flat)  # (B, N, d)
        K = self.k_proj(f_flat)  # (B, N, d)
        V = self.v_proj(
            f_flat
        )  # (B, N, C) - Assuming we want to reconstruct C channels

        # Attention Matrix
        # (B, N, d) @ (B, d, N) -> (B, N, N)
        attn = torch.bmm(Q, K.transpose(1, 2))
        attn = attn / (self.embed_dim**0.5)
        attn = F.softmax(attn, dim=-1)  # (B, N, N)

        # Token Selection (Top-k)
        # "Calculate total attention value for each Key Token"
        # Sum over Query dimension (dim 1)? Or sum over Key dimension (dim 2)?
        # Usually "importance of a key" is how much it is attended to by all queries.
        # So sum over dim 1 (Queries).
        key_importance = attn.sum(dim=1)  # (B, N)

        # Select Top-k tokens (pixels)
        # k is likely a percentage or a fixed number.
        # If k is small integer like 2, it's too small for pixels.
        # The paper might mean k% or k channels?
        # "Keep top k tokens" usually implies spatial tokens.
        # Let's assume k is a ratio, e.g., 0.5 * N.
        # Or if the user passed k=2 in the class init, maybe it's for the DTConv branches?
        # For MDAM, let's assume we keep a fraction of tokens.
        # But the prompt says "k tokens".
        # Let's implement a flexible k. If k < 1, treat as ratio. If k > 1, treat as count.

        N = h * w
        k_val = self.k
        if k_val < 1:
            k_val = int(N * k_val)
        k_val = min(k_val, N)

        topk_vals, topk_indices = torch.topk(key_importance, k_val, dim=1)

        # Create a mask for tokens
        # We want to mask out columns in the attention matrix corresponding to unimportant keys?
        # Or mask the Value tokens?
        # "Output = Concat(A'_t * V_t) * W"
        # Usually we mask the attention weights so unimportant keys don't contribute.

        mask = torch.zeros_like(key_importance)
        mask.scatter_(1, topk_indices, 1.0)  # (B, N)
        mask = mask.unsqueeze(1)  # (B, 1, N) - broadcast over queries

        # Attention Correction
        # "Multiply by correction coefficient alpha" - let's assume alpha=1 for selected, 0 for others (hard mask)
        # Or maybe re-normalize?
        # Let's apply the mask to attention matrix
        attn_masked = attn * mask

        # Apply to V
        # (B, N, N) @ (B, N, C) -> (B, N, C)
        out = torch.bmm(attn_masked, V)

        # Output projection
        out = self.out_proj(out)  # (B, N, C)

        # Reshape back to (B, C, H, W)
        out = out.transpose(1, 2).view(b, c, h, w)

        return out


class LCNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.stride = stride

        # Input transformation
        # 1x1 Conv + DWConv
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                stride=stride,
                padding=1,
                groups=out_channels,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Dual Path
        # DTConv
        self.dtconv = DynamicThresholdConv(out_channels)
        # MDAM
        self.mdam = MultipathDynamicAttention(out_channels)

        # Output transformation
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Shortcut
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        x = self.input_conv(x)

        # Dual Path
        out_dt = self.dtconv(x)
        out_mdam = self.mdam(x)

        # Star Operation (Element-wise product)
        x = out_dt * out_mdam

        x = self.output_conv(x)

        return x + residual


class LCNet(nn.Module):
    def __init__(self, num_classes: int = 10, variant: str = "tiny"):
        super().__init__()

        # Config based on variant
        if variant == "tiny":
            stem_filters = 24
            stage_configs = [
                (32, 2),  # Stage 1: (channels, layers)
                (64, 4),  # Stage 2
                (128, 8),  # Stage 3
                (192, 2),  # Stage 4
            ]
        elif variant == "small":
            stem_filters = 32
            stage_configs = [(48, 2), (96, 4), (192, 12), (256, 2)]
        else:  # base
            stem_filters = 32
            stage_configs = [(64, 4), (128, 6), (256, 16), (256, 4)]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_filters, 3, stride=2, padding=1),
            nn.BatchNorm2d(stem_filters),
            nn.ReLU(inplace=True),
        )

        # Stages
        self.stages = nn.ModuleList()
        in_ch = stem_filters

        for out_ch, num_layers in stage_configs:
            layers = []
            # First layer of each stage usually handles stride/downsampling if needed
            # The paper says "Resolution halves" at each stage.
            # Stem (224->112). Stage 1 (56?).
            # Standard ResNet: Stem /2. Stage 1 /1. Stage 2 /2. Stage 3 /2. Stage 4 /2.
            # LCNet table: Stem 224. Stage 1 56. Stage 2 28. Stage 3 14. Stage 4 7.
            # So Stem is /4? Or Stem /2 then MaxPool?
            # "Stem: 3x3 Conv (stride=2)". That's /2 (112).
            # Stage 1 is 56. So we need another /2 before Stage 1.
            # Let's assume the first block of each stage (except maybe stage 1?) does stride=2.
            # Wait, 112 -> 56 is /2.

            for i in range(num_layers):
                stride = 2 if i == 0 else 1
                # Special case: If it's the very first stage, and we are at 112, we need to go to 56.
                # So stride=2 is correct for first block of ALL stages to match the resolution table.

                layers.append(LCNetBlock(in_ch, out_ch, stride=stride))
                in_ch = out_ch

            self.stages.append(nn.Sequential(*layers))

        # Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        for stage in self.stages:
            x = stage(x)

        x = self.gap(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x
