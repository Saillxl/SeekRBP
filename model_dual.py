import torch
import torch.nn as nn
from model import Encoder 


class Trainable1DEncoder(nn.Module):
    def __init__(
        self,
        d_in_1d: int,
        embed_size: int,
        heads: int,
        forward_expansion: int,
        dropout: float,
        max_length: int,
        device: torch.device,
        out_dim: int = 256,
        num_layers: int = 1,
    ):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size=d_in_1d,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            device=device,
            forward_expansion=forward_expansion,
            dropout=dropout,
            max_length=max_length,
        )
        self.proj = nn.Linear(embed_size, out_dim)
        self.out_dim = out_dim

    def forward(self, x_1d: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(x_1d, mask=None)  # [B, L1, E]
        pooled = tokens.mean(dim=1)             # [B, E]
        return self.proj(pooled)                # [B, out_dim]


class Trainable3DEncoder(nn.Module):
    def __init__(
        self,
        d_in_3d: int,
        embed_size: int,
        heads: int,
        forward_expansion: int,
        dropout: float,
        max_length: int,
        device: torch.device,
        out_dim: int = 256,
        num_layers: int = 1,
    ):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size=d_in_3d,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            device=device,
            forward_expansion=forward_expansion,
            dropout=dropout,
            max_length=max_length,
        )
        self.proj = nn.Linear(embed_size, out_dim)
        self.out_dim = out_dim

    def forward(self, x_3d: torch.Tensor) -> torch.Tensor:
        """
        x_3d: [B, L3, D3]
        return emb_3d: [B, out_dim]
        """
        tokens = self.encoder(x_3d, mask=None)  # [B, L3, E]
        pooled = tokens.mean(dim=1)             # [B, E]
        return self.proj(pooled)                # [B, out_dim]


class LowRankBilinear(nn.Module):
    """Low-rank bilinear interaction: h = (U1 z1) ⊙ (U3 z3), then linear projection"""
    def __init__(self, d_in, d_out, rank=64):
        super().__init__()
        self.u1 = nn.Linear(d_in, rank, bias=False)
        self.u3 = nn.Linear(d_in, rank, bias=False)
        self.proj = nn.Linear(rank, d_out)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.u1.weight)
        nn.init.xavier_uniform_(self.u3.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, z1, z3):           # [B,d], [B,d]
        h = self.u1(z1) * self.u3(z3)    # [B,rank]
        return self.proj(h)               # [B,d_out]


class AEFM(nn.Module):
    """
    - Gated Additive Path：Learn α1 and α3, and perform additive fusion (who is more important)
    - Low-rank Bilinear Path：Learn multiplicative interaction
    - Mix Gate β：Adaptively select the ratio of "additive" vs "multiplicative"
    - Output: [B, d_out]
    """
    def __init__(self, d_in: int, d_out: int, rank: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_in, self.d_out = d_in, d_out

        # Gated Additive Path: Learn α1 and α3, and perform additive fusion (who is more important)
        self.gate = nn.Sequential(
            nn.LayerNorm(d_in * 2),
            nn.Linear(d_in * 2, d_in),
            nn.SiLU(),
            nn.Linear(d_in, 2),   # -> [α1, α3]
        )

        # Additive Path: Linear [α1*z1 ; α3*z3] to d_out
        self.add_proj = nn.Sequential(
            nn.LayerNorm(d_in * 2),
            nn.Dropout(dropout),
            nn.Linear(d_in * 2, d_out),
        )

        # Learn multiplicative interaction
        self.bilinear = LowRankBilinear(d_in, d_out, rank=rank)

        # Mix Gate β: Control the ratio of "multiplicative"
        self.mix_gate = nn.Sequential(
            nn.LayerNorm(d_out * 3),
            nn.Linear(d_out * 3, d_out),
        )

        self.reset_parameters()

    def reset_parameters(self):
        # add_proj, mix_gate are initialized internally; here only do a slight bias
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, z1: torch.Tensor, z3: torch.Tensor) -> torch.Tensor:
        """
        z1, z3: [B, d_in]
        return: fused [B, d_out]
        """
        x = torch.cat([z1, z3], dim=-1)           # [B, 2d]
        g = self.gate(x).softmax(dim=-1)          # [B, 2] -> α1, α3
        a1, a3 = g.unbind(-1)                     # [B], [B]
        add_in = torch.cat([a1.unsqueeze(-1) * z1,
                            a3.unsqueeze(-1) * z3], dim=-1)     # [B, 2d]
        add_out = self.add_proj(add_in)                           # [B, d_out]

        mul_out = self.bilinear(z1, z3)                           # [B, d_out]
        gate_in = torch.cat([add_out, mul_out, (add_out - mul_out).abs()], dim=-1)
        beta = torch.sigmoid(self.mix_gate(gate_in))
        fused = beta * add_out + (1.0 - beta) * mul_out           # Mixed by channel
        return fused


class DualBranchClassifier(nn.Module):
    def __init__(self, enc1d, enc3d, num_classes=2, fusion_hidden=256, dropout=0.2,
                 fusion_out_dim=None, rank=64):
        super().__init__()
        self.enc1d = enc1d
        self.enc3d = enc3d
        d1 = enc1d.out_dim
        d3 = enc3d.out_dim
        assert d1 == d3, "It is recommended that the out_dim values for the two paths be the same. If they are different, you can modify AEFM to accept d1, d3 → d_out."

        d = d1
        self.fusion = AEFM(d_in=d, d_out=fusion_out_dim or d, rank=rank, dropout=dropout)
        d_fused = fusion_out_dim or d

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_fused),
            nn.Dropout(dropout),
            nn.Linear(d_fused, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, x_1d, x_3d):
        z1 = self.enc1d(x_1d)  # [B,d]
        z3 = self.enc3d(x_3d)  # [B,d]
        z  = self.fusion(z1, z3)
        return self.classifier(z)