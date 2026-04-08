import math

import torch
import torch.nn.functional as F
from torch import nn


# ─────────────────────────────────────────────────────────────
# 1. Activation Swish
# ─────────────────────────────────────────────────────────────
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# ─────────────────────────────────────────────────────────────
# 2. Embedding temporel (sinusoidal)
# ─────────────────────────────────────────────────────────────
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half = dim // 4
        self.register_buffer(
            "frequencies", torch.exp(torch.arange(half) * -(math.log(10000) / half))
        )
        self.mlp = nn.Sequential(
            nn.Linear(half * 2, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        emb = t[:, None] * self.frequencies[None, :]  # (B, half)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, half*2)
        return self.mlp(emb)  # (B, dim)


# ─────────────────────────────────────────────────────────────
# 3. Bloc résiduel avec conditionnement temps
# ─────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_emb_dim: int,
        groups: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_ch)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.shortcut = (
            nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
        )
        self.act = Swish()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.shortcut(x)


# ─────────────────────────────────────────────────────────────
# 4. Attention (optionnel, à 14×14 pour 28×28)
# ─────────────────────────────────────────────────────────────
class AttentionBlock(nn.Module):
    def __init__(self, ch: int, groups: int = 8, n_heads: int = 1):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.n_heads = n_heads

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None) -> torch.Tensor:
        B, C, H, W = x.shape
        _ = t_emb  # unused, kept for API consistency

        qkv = self.qkv(self.norm(x))  # (B, 3*C, H, W)
        q, k, v = qkv.chunk(3, dim=1)

        # Flatten spatial dims
        q, k, v = [t.flatten(2).transpose(1, 2) for t in (q, k, v)]  # (B, HW, C)

        # Multi-head attention
        scale = (C // self.n_heads) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = attn @ v

        # Reshape back
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj(out)


# ─────────────────────────────────────────────────────────────
# 5. Downsample / Upsample
# ─────────────────────────────────────────────────────────────
class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb=None):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x, t_emb=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ─────────────────────────────────────────────────────────────
# 6. UNet principal
# ─────────────────────────────────────────────────────────────
class UNet(nn.Module):
    """
    UNet pour DDPM - entrée: (B, 3, 28, 28), sortie: (B, 3, 28, 28)
    Architecture adaptée de Ho et al. 2020 [[11]][[29]]
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4),  # 28→14→7→7→14→28
        attn_resolutions: tuple = (14,),  # attention à 14×14
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        time_emb_dim: int = None,
    ):
        super().__init__()
        time_emb_dim = time_emb_dim or base_channels * 4

        # --- Embedding temporel ---
        self.time_emb = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            Swish(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # --- Conv d'entrée ---
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # --- Downsampling ---
        self.down_modules = nn.ModuleList()
        ch = base_channels
        skips_ch = [ch]

        for i, mult in enumerate(channel_mults):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.down_modules.append(
                    ResidualBlock(ch, out_ch, time_emb_dim, dropout=dropout)
                )
                ch = out_ch
                if ch in [c * base_channels for c in attn_resolutions]:
                    self.down_modules.append(AttentionBlock(ch))
                skips_ch.append(ch)
            if i < len(channel_mults) - 1:
                self.down_modules.append(Downsample(ch))
                skips_ch.append(ch)

        # --- Middle ---
        self.middle = nn.ModuleList(
            [
                ResidualBlock(ch, ch, time_emb_dim, dropout=dropout),
                AttentionBlock(ch),
                ResidualBlock(ch, ch, time_emb_dim, dropout=dropout),
            ]
        )

        # --- Upsampling ---
        self.up_modules = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = ch // mult if i > 0 else ch
            for _ in range(num_res_blocks + 1):
                skip_ch = skips_ch.pop()
                self.up_modules.append(
                    ResidualBlock(ch + skip_ch, out_ch, time_emb_dim, dropout=dropout)
                )
                ch = out_ch
                if ch in [c * base_channels for c in attn_resolutions]:
                    self.up_modules.append(AttentionBlock(ch))
            if i > 0:
                self.up_modules.append(Upsample(ch))

        # --- Sortie ---
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
        self.act = Swish()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 28, 28) - image bruitée à l'étape t
            t: (B,) - timestep (entiers 0..T-1)
        Returns:
            epsilon_pred: (B, 3, 28, 28) - bruit prédit
        """
        t_emb = self.time_emb(t)  # (B, time_emb_dim)

        # Entry
        x = self.input_conv(x)
        skips = [x]

        # Down
        for m in self.down_modules:
            if isinstance(m, (Downsample, AttentionBlock)):
                x = m(x)
            else:
                x = m(x, t_emb)
            skips.append(x)

        # Middle
        for m in self.middle:
            x = m(x, t_emb) if isinstance(m, ResidualBlock) else m(x)

        # Up
        for m in self.up_modules:
            if isinstance(m, Upsample):
                x = m(x)
            elif isinstance(m, AttentionBlock):
                x = m(x)
            else:
                x = torch.cat([x, skips.pop()], dim=1)
                x = m(x, t_emb)

        # Output
        return self.out_conv(self.act(self.out_norm(x)))


if __name__ == "__main__":
    # Instanciation pour (3, 28, 28)
    model = UNet(in_channels=3, out_channels=3, base_channels=64)

    # Forward pass
    x_noisy = torch.randn(4, 3, 28, 28)  # batch de 4 images
    timesteps = torch.randint(0, 1000, (4,))  # t aléatoires
    epsilon_pred = model(x_noisy, timesteps)  # → (4, 3, 28, 28)

    # Perte DDPM simplifiée (L_simple)
    epsilon_true = torch.randn_like(epsilon_pred)
    loss = F.mse_loss(epsilon_pred, epsilon_true)
    print("Loss:", loss.item())
