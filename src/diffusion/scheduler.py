"""
scheduler.py
------------
Noise scheduler pour le processus de diffusion (DDPM).

Le scheduler definit comment le bruit est ajoute progressivement
a l'image au cours du processus forward, et comment il est retire
pendant le sampling (processus reverse).

Terminologie :
  - beta_t : variance du bruit ajoute au pas t
  - alpha_t : 1 - beta_t
  - alpha_bar_t : produit cumulatif des alpha_t (= combien de signal reste)
  - On peut echantillonner directement x_t a partir de x_0 grace a alpha_bar
"""

import torch
import torch.nn as nn
import numpy as np


class LinearNoiseScheduler(nn.Module):
    """
    Scheduler lineaire : beta croit lineairement de beta_start a beta_end.

    C'est le schedule original du papier DDPM (Ho et al. 2020).
    Simple et efficace, meme si le schedule cosine est parfois meilleur.

    Pre-calcule toutes les constantes necessaires pour l'entrainement
    et le sampling, ce qui evite de les recalculer a chaque iteration.
    """

    def __init__(self, n_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.n_timesteps = n_timesteps

        # Schedule lineaire des betas
        betas = torch.linspace(beta_start, beta_end, n_timesteps)

        # Pre-calcul des constantes derivees
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.tensor([1.0]), alpha_bar[:-1]])

        # Variance du processus reverse (pour le sampling)
        # sigma_t^2 = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

        # Enregistrer comme buffers (pas des parametres entrainables,
        # mais doivent etre deplaces avec .to(device))
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('alpha_bar_prev', alpha_bar_prev)
        self.register_buffer('sqrt_alpha_bar', torch.sqrt(alpha_bar))
        self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1.0 - alpha_bar))
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('sqrt_recip_alpha', torch.sqrt(1.0 / alphas))
        self.register_buffer(
            'posterior_mean_coef',
            betas / torch.sqrt(1.0 - alpha_bar)
        )

    def add_noise(self, x_0, t, noise=None):
        """
        Processus forward : ajoute du bruit a x_0 pour obtenir x_t.

        q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

        Grace a la propriete de reparametrisation, on peut echantillonner
        directement x_t sans passer par tous les pas intermediaires.

        Args:
            x_0: images propres (B, C, H, W)
            t: timesteps (B,)
            noise: bruit optionnel (si None, genere du bruit gaussien)

        Returns:
            x_t: images bruitees
            noise: le bruit ajoute (pour calculer la perte)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]

        x_t = sqrt_ab * x_0 + sqrt_one_minus_ab * noise
        return x_t, noise

    def reverse_step(self, x_t, t, predicted_noise):
        """
        Un pas du processus reverse (sampling).

        p(x_{t-1} | x_t) = N(mu_theta, sigma_t^2 * I)

        ou mu_theta est calcule a partir de la prediction du bruit
        par le reseau de debruitage.

        Args:
            x_t: image bruitee au pas t
            t: timestep (entier)
            predicted_noise: bruit predit par le U-Net

        Returns:
            x_{t-1}: image un peu moins bruitee
        """
        sqrt_recip_alpha = self.sqrt_recip_alpha[t]
        posterior_mean_coef = self.posterior_mean_coef[t]

        # Moyenne du posterior
        mean = sqrt_recip_alpha * (x_t - posterior_mean_coef * predicted_noise)

        if t > 0:
            # Ajouter du bruit (sauf au dernier pas)
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(self.posterior_variance[t])
            return mean + sigma * noise
        else:
            return mean
