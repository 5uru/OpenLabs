from typing import Optional, Tuple

import numpy as np
import torch
from unet import UNet


# ─────────────────────────────────────────────────────────────
# 1. Configuration du schedule de bruit (linear)
# ─────────────────────────────────────────────────────────────
class DiffusionSchedule:
    """
    Schedule de bruit DDPM - β_t linéaire de 1e-4 à 0.02
    """

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)  # (T,)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # α̅_t

        # Coefficients pour le sampling (reverse process)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # Variances pour le reverse process (posterior)
        self.posterior_variance = (
            self.betas[1:]  # Use betas[1:] instead of self.betas
            * (1.0 - self.alphas_cumprod[:-1])
            / (1.0 - self.alphas_cumprod[1:])
        )

        self.posterior_variance = torch.cat([self.betas[0:1], self.posterior_variance])
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        # Coeffs pour q(x_{t-1} | x_t, x_0)
        self.posterior_mean_coef1 = (
            self.betas[1:]
            * torch.sqrt(self.alphas_cumprod[:-1])
            / (1.0 - self.alphas_cumprod[1:])
        )

        self.posterior_mean_coef1 = torch.cat(
            [self.betas[0:1], self.posterior_mean_coef1]
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod[:-1])
            * torch.sqrt(self.alphas[1:])
            / (1.0 - self.alphas_cumprod[1:])
        )

        self.posterior_mean_coef2 = torch.cat(
            [self.betas[0:1], self.posterior_mean_coef2]
        )

    def to(self, device):
        """Déplace tous les buffers sur le device spécifié"""
        for attr in dir(self):
            val = getattr(self, attr)
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.to(device))
        return self


# ─────────────────────────────────────────────────────────────
# 2. Forward process : ajouter du bruit à x_0
# ─────────────────────────────────────────────────────────────
def q_sample(
    schedule: DiffusionSchedule,
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Échantillonne x_t ~ q(x_t | x_0) = N(x_t; √α̅_t x_0, (1-α̅_t)I)

    Args:
        x0: (B, C, H, W) - image originale
        t: (B,) - timesteps (0 à T-1)
        noise: bruit optionnel, sinon généré

    Returns:
        xt: image bruitée à l'étape t
        noise: le bruit utilisé (pour le calcul de la perte)
    """
    if noise is None:
        noise = torch.randn_like(x0)

    # Indexer les coefficients par timestep
    sqrt_alphas_cumprod_t = schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = schedule.sqrt_one_minus_alphas_cumprod[t].view(
        -1, 1, 1, 1
    )

    # Reparameterization trick
    xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
    return xt, noise


# ─────────────────────────────────────────────────────────────
# 3. Reverse process : prédire x_{t-1} depuis x_t
# ─────────────────────────────────────────────────────────────
def p_sample(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    xt: torch.Tensor,
    t: torch.Tensor,
    clip_denoised: bool = True,
) -> torch.Tensor:
    """
    Échantillonne x_{t-1} ~ p_θ(x_{t-1} | x_t)

    Utilise la prédiction du bruit ε_θ(x_t, t) pour reconstruire x_0,
    puis applique la formule de la posterior q(x_{t-1}|x_t, x_0).
    """
    B = xt.shape[0]

    # 1. Prédire le bruit
    epsilon_pred = model(xt, t)  # (B, C, H, W)

    # 2. Estimer x_0 depuis x_t et ε_pred
    # x_0 = (x_t - √(1-α̅_t)·ε) / √α̅_t
    sqrt_recip_alphas_cumprod_t = schedule.sqrt_recip_alphas_cumprod[t].view(
        -1, 1, 1, 1
    )
    sqrt_recipm1_alphas_cumprod_t = schedule.sqrt_recipm1_alphas_cumprod[t].view(
        -1, 1, 1, 1
    )

    x0_pred = (
        sqrt_recip_alphas_cumprod_t * xt - sqrt_recipm1_alphas_cumprod_t * epsilon_pred
    )

    if clip_denoised:
        x0_pred = x0_pred.clamp(-1.0, 1.0)

    # 3. Calculer la moyenne de la posterior q(x_{t-1}|x_t, x_0)
    posterior_mean_coef1_t = schedule.posterior_mean_coef1[t].view(-1, 1, 1, 1)
    posterior_mean_coef2_t = schedule.posterior_mean_coef2[t].view(-1, 1, 1, 1)

    mean = posterior_mean_coef1_t * x0_pred + posterior_mean_coef2_t * xt

    # 4. Ajouter du bruit si t > 0
    if t[0].item() == 0:
        return mean  # pas de bruit à t=0
    else:
        posterior_log_variance_t = schedule.posterior_log_variance_clipped[t].view(
            -1, 1, 1, 1
        )
        noise = torch.randn_like(xt)
        return mean + torch.exp(0.5 * posterior_log_variance_t) * noise


# ─────────────────────────────────────────────────────────────
# 4. Boucle de sampling complète
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def p_sample_loop(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    shape: Tuple[int, int, int, int],
    device: torch.device,
    clip_denoised: bool = True,
    progress: bool = False,
) -> torch.Tensor:
    """
    Génère des images depuis du bruit pur : x_T → x_0

    Args:
        shape: (B, C, H, W)
        progress: affiche une barre de progression si True
    """
    B, C, H, W = shape
    img = torch.randn(shape, device=device)  # x_T ~ N(0, I)

    indices = list(reversed(range(schedule.T)))
    if progress:
        from tqdm import tqdm

        indices = tqdm(indices)

    for i in indices:
        t = torch.full((B,), i, device=device, dtype=torch.long)
        img = p_sample(model, schedule, img, t, clip_denoised=clip_denoised)

    return img


if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = DiffusionSchedule(T=1000).to(device)
    model = UNet(in_channels=3, out_channels=3, base_channels=64).to(device)

    # ─── Forward pass (entraînement) ───
    x0 = torch.randn(4, 3, 28, 28).to(device)  # tes images normalisées [-1, 1]
    t = torch.randint(0, schedule.T, (4,), device=device)
    xt, noise = q_sample(schedule, x0, t)

    # Prédiction et perte
    epsilon_pred = model(xt, t)
    loss = torch.nn.functional.mse_loss(epsilon_pred, noise)

    # ─── Reverse process (génération) ───
    model.eval()
    generated = p_sample_loop(
        model, schedule, shape=(4, 3, 28, 28), device=device, progress=True
    )
    print("Génération terminée, shape:", generated.shape)
