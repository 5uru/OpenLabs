import math

import torch
import torch.nn as nn


def exists(x):
    return x is not None


# 🕐 Time Embedding (standard DDPM)
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        x = t[:, None] * freqs[None, :]
        return torch.cat((x.sin(), x.cos()), dim=-1)


# 🧱 ResBlock avec conditionnement temporel (FiLM: Scale & Shift)
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim
            else None
        )

        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, dim), nn.Conv2d(dim, dim_out, 3, padding=1), nn.SiLU()
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, dim_out),
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
            nn.SiLU(),
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, t_emb=None):
        h = self.block1(x)
        if exists(self.mlp) and exists(t_emb):
            scale, shift = self.mlp(t_emb).chunk(2, dim=1)
            # FiLM: h = h * (scale + 1) + shift
            h = h * (scale[:, :, None, None] + 1) + shift[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


# 🕸️ U-Net explicite & stable pour (3, 28, 28)
class Unet(nn.Module):
    def __init__(self, dim=32, channels=3, time_emb_dim=128):
        super().__init__()
        # 1️⃣ Time MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.init_conv = nn.Conv2d(channels, dim, 3, padding=1)

        # 2️️ DOWN PATH (28→14→7)
        self.d1_b1 = ResnetBlock(dim, dim * 2, time_emb_dim=time_emb_dim)
        self.d1_b2 = ResnetBlock(dim * 2, dim * 2, time_emb_dim=time_emb_dim)
        self.d1_down = nn.Conv2d(dim * 2, dim * 2, 4, 2, 1)  # 28→14

        self.d2_b1 = ResnetBlock(dim * 2, dim * 4, time_emb_dim=time_emb_dim)
        self.d2_b2 = ResnetBlock(dim * 4, dim * 4, time_emb_dim=time_emb_dim)
        self.d2_down = nn.Conv2d(dim * 4, dim * 4, 4, 2, 1)  # 14→7

        # 3️⃣ MID (7×7)
        self.mid1 = ResnetBlock(dim * 4, dim * 4, time_emb_dim=time_emb_dim)
        self.mid2 = ResnetBlock(dim * 4, dim * 4, time_emb_dim=time_emb_dim)

        # 4️⃣ UP PATH (7→14→28)
        self.u2_up = nn.ConvTranspose2d(dim * 4, dim * 4, 4, 2, 1)  # 7→14
        # Attention aux canaux après concaténation avec les skips
        self.u2_b1 = ResnetBlock(
            dim * 4 + dim * 4, dim * 2, time_emb_dim=time_emb_dim
        )  # 128+128 → 64
        self.u2_b2 = ResnetBlock(
            dim * 2 + dim * 4, dim * 2, time_emb_dim=time_emb_dim
        )  # 64+128  → 64

        self.u1_up = nn.ConvTranspose2d(dim * 2, dim * 2, 4, 2, 1)  # 14→28
        self.u1_b1 = ResnetBlock(
            dim * 2 + dim * 2, dim, time_emb_dim=time_emb_dim
        )  # 64+64   → 32
        self.u1_b2 = ResnetBlock(
            dim + dim * 2, dim, time_emb_dim=time_emb_dim
        )  # 32+64   → 32

        # 5️⃣ OUTPUT
        self.final_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim // 2, channels, 1),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)  # (B, 32, 28, 28)

        # 🔽 DOWN
        x = self.d1_b1(x, t_emb)
        s1_1 = x
        x = self.d1_b2(x, t_emb)
        s1_2 = x
        x = self.d1_down(x)  # (B, 64, 14, 14)

        x = self.d2_b1(x, t_emb)
        s2_1 = x
        x = self.d2_b2(x, t_emb)
        s2_2 = x
        x = self.d2_down(x)  # (B, 128, 7, 7)

        # 🔄 MID
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)  # (B, 128, 7, 7)

        # 🔼 UP (niveau 2 → 1)
        x = self.u2_up(x)  # (B, 128, 14, 14)
        x = torch.cat([x, s2_2], dim=1)
        x = self.u2_b1(x, t_emb)  # 256 → 64
        x = torch.cat([x, s2_1], dim=1)
        x = self.u2_b2(x, t_emb)  # 192 → 64

        # 🔼 UP (niveau 1 → 0)
        x = self.u1_up(x)  # (B, 64, 28, 28)
        x = torch.cat([x, s1_2], dim=1)
        x = self.u1_b1(x, t_emb)  # 128 → 32
        x = torch.cat([x, s1_1], dim=1)
        x = self.u1_b2(x, t_emb)  # 96  → 32

        return self.final_conv(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet(dim=32, channels=3, time_emb_dim=128).to(device)

    x = torch.randn(4, 3, 28, 28, device=device)
    t = torch.randint(0, 1000, (4,), device=device)

    out = model(x, t)
    assert out.shape == (4, 3, 28, 28), f"❌ Shape mismatch: {out.shape}"

    # Vérifie la rétropropagation
    loss = out.sum()
    loss.backward()

    print("✅ U-Net prêt:")
    print(f"   Entrée : {list(x.shape)} + t")
    print(f"   Sortie : {list(out.shape)}")
    print("   Backprop : OK")
