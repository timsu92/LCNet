import torch
import torch.nn as nn


class GroupRationalActivation(nn.Module):
    """
    Section 4.2 & 4.3: Group-Rational KAN (GR-KAN) Activation
    Paper Eq (12): F(x) = P(x) / Q(x)
    Paper Eq (17): GR-KAN(x) = Linear(Group_Rational(x)) -> We implement the Group_Rational part here.
    """

    def __init__(self, in_channels: int, groups: int = 8, m: int = 5, n: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.m = m  # Numerator order [cite: 1045]
        self.n = n  # Denominator order [cite: 1045]

        # Ensure channels are divisible by groups
        assert in_channels % groups == 0, "in_channels must be divisible by groups"

        # Coefficient initialization
        # a_coeffs (Numerator): Shape (1, Groups, 1, 1, m+1) - Shared per group
        self.a_coeffs = nn.Parameter(torch.Tensor(1, groups, 1, 1, m + 1))

        # b_coeffs (Denominator): Shape (1, Groups, 1, 1, n) - Shared per group
        # Note: Paper suggests b_n can be shared among ALL groups for better performance[cite: 1106],
        # but here we follow standard group sharing for flexibility.
        self.b_coeffs = nn.Parameter(torch.Tensor(1, groups, 1, 1, n))

        # w_scale: Unique scalar for EACH channel (not just group) [cite: 1060]
        # Shape: (1, C, 1, 1)
        self.w_scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialization strategy as described in Section 4.4 [cite: 1152]
        # Initializing to approximate Identity function as a starting point is common for deep networks
        nn.init.normal_(self.a_coeffs, mean=0.0, std=0.1)
        nn.init.normal_(self.b_coeffs, mean=0.0, std=0.1)

        # Set a_1 to 1.0 and others small to approximate identity: F(x) ≈ x
        with torch.no_grad():
            self.a_coeffs[:, :, :, :, 1] = 1.0
            self.a_coeffs[:, :, :, :, 0] = 0.0  # Bias term of numerator

        nn.init.normal_(self.w_scale, mean=1.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        b, c, h, w = x.shape

        # Reshape for group operation: (B, Groups, Channels/Groups, H, W)
        x_grouped = x.view(b, self.groups, c // self.groups, h, w)

        # ===== P(x) Calculation (Numerator) =====
        # Using Horner's method: a0 + x(a1 + x(a2 + ...)) [cite: 1041]
        # Polynomial: a_0 + a_1*x + a_2*x^2 + ... + a_m*x^m
        # Coefficients shape needs to match x_grouped for broadcasting
        # self.a_coeffs: (1, Groups, 1, 1, m+1)
        # We need to expand to: (1, Groups, 1, 1, 1) for each coefficient to broadcast with (B, Groups, C/G, H, W)

        P = self.a_coeffs[:, :, :, :, self.m].unsqueeze(-1)  # (1, G, 1, 1, 1)
        for i in range(self.m - 1, -1, -1):
            coeff = self.a_coeffs[:, :, :, :, i].unsqueeze(-1)  # (1, G, 1, 1, 1)
            P = P * x_grouped + coeff

        # ===== Q(x) Calculation (Denominator) =====
        # Denominator: 1 + |b_1*x + b_2*x^2 + ... + b_n*x^n|
        # Note: b starts from degree 1 (no b_0 term)
        Q_poly = self.b_coeffs[:, :, :, :, self.n - 1].unsqueeze(-1)  # (1, G, 1, 1, 1)
        for i in range(self.n - 2, -1, -1):
            coeff = self.b_coeffs[:, :, :, :, i].unsqueeze(-1)  # (1, G, 1, 1, 1)
            Q_poly = Q_poly * x_grouped + coeff

        # Multiply by x to account for the fact that b starts from x^1
        Q_poly = Q_poly * x_grouped

        # Final denominator: 1 + |Q_poly|
        Q = 1.0 + torch.abs(Q_poly)

        # ===== Rational Function: F(x) = P(x) / Q(x) =====
        y = P / (Q + 1e-8)  # Add epsilon for numerical stability

        # Reshape back to (B, C, H, W)
        y = y.view(b, c, h, w)

        # Apply channel-wise scaling w [cite: 1060]
        return y * self.w_scale
