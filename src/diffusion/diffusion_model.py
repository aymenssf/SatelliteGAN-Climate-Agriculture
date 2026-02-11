"""
diffusion_model.py
------------------
Implementation du Denoising Diffusion Probabilistic Model (DDPM).

Le DDPM est un modele generatif en deux phases :
  1. Forward process : on ajoute progressivement du bruit gaussien
     a une image propre, en T pas, jusqu'a obtenir du bruit pur.
  2. Reverse process : on part de bruit pur et on debruite
     progressivement avec un reseau de neurones (U-Net), en T pas,
     pour generer une image.

Le reseau est entraine pour predire le bruit ajoute a chaque pas.
La perte est simplement un MSE entre le bruit reel et le bruit predit.

Reference : Ho, J., Jain, A., & Abbeel, P. (2020)
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from src.diffusion.unet import UNet
from src.diffusion.scheduler import LinearNoiseScheduler
from src.config import DEVICE, DIFFUSION


class DDPM(nn.Module):
    """
    Modele de diffusion DDPM complet.

    Combine le U-Net (reseau de debruitage) et le scheduler
    (gestion du bruit) dans un seul module.
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or DIFFUSION

        # U-Net de debruitage
        self.unet = UNet(
            in_channels=3,
            base_channels=cfg['base_channels'],
            channel_mults=tuple(cfg['channel_mults']),
            n_res_blocks=cfg['n_res_blocks'],
            attention_levels=cfg['attention_levels'],
        )

        # Scheduler de bruit
        self.scheduler = LinearNoiseScheduler(
            n_timesteps=cfg['n_timesteps'],
            beta_start=cfg['beta_start'],
            beta_end=cfg['beta_end'],
        )

        self.n_timesteps = cfg['n_timesteps']
        self.sampling_steps = cfg.get('sampling_steps', 50)

    def forward(self, x_0):
        """
        Pas d'entrainement : calcule la perte de debruitage.

        1. Echantillonne un timestep aleatoire pour chaque image du batch
        2. Ajoute le bruit correspondant
        3. Predit le bruit avec le U-Net
        4. Retourne le MSE entre bruit reel et bruit predit

        Args:
            x_0: batch d'images propres (B, C, H, W) dans [-1, 1]

        Returns:
            loss: perte MSE scalaire
        """
        batch_size = x_0.shape[0]

        # Echantillonner des timesteps uniformement
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x_0.device)

        # Ajouter du bruit
        x_t, noise = self.scheduler.add_noise(x_0, t)

        # Predire le bruit
        predicted_noise = self.unet(x_t, t)

        # Perte simple : MSE entre bruit reel et predit
        loss = nn.functional.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def sample(self, n_samples, image_size=64, device=None):
        """
        Genere des images par le processus reverse de diffusion.

        Part de bruit gaussien pur et debruite progressivement
        en utilisant le U-Net entraine.

        Args:
            n_samples: nombre d'images a generer
            image_size: taille spatiale des images
            device: device (defaut: DEVICE de config)

        Returns:
            images: tensor (n_samples, 3, H, W) dans [-1, 1]
        """
        device = device or DEVICE

        # Partir de bruit pur
        x = torch.randn(n_samples, 3, image_size, image_size, device=device)

        # Debruiter pas a pas (du plus bruite au plus propre)
        for t in tqdm(reversed(range(self.n_timesteps)), total=self.n_timesteps,
                      desc="Sampling"):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            predicted_noise = self.unet(x, t_batch)
            x = self.scheduler.reverse_step(x, t, predicted_noise)

        # Clamp dans [-1, 1]
        x = x.clamp(-1, 1)
        return x

    @torch.no_grad()
    def sample_fast(self, n_samples, image_size=64, device=None):
        """
        Sampling accelere avec DDIM (deterministe, meilleure qualite).

        Utilise DDIM au lieu de DDPM pour un sampling rapide :
        100 steps DDIM ~ 1000 steps DDPM en qualite.

        Args:
            n_samples: nombre d'images a generer
            image_size: taille spatiale des images
            device: device

        Returns:
            images: tensor (n_samples, 3, H, W) dans [-1, 1]
        """
        return self.sample_ddim(n_samples, image_size, device,
                                num_steps=self.sampling_steps, eta=0.0)

    @torch.no_grad()
    def sample_ddim(self, n_samples, image_size=64, device=None,
                    num_steps=100, eta=0.0):
        """
        Sampling DDIM (Song et al., 2021).

        DDIM est deterministe (eta=0) et produit des images de meilleure
        qualite que le DDPM stochastique a nombre de pas egal.

        Args:
            n_samples: nombre d'images a generer
            image_size: taille spatiale des images
            device: device
            num_steps: nombre de pas de sampling (100 ~ 1000 DDPM)
            eta: stochasticite (0=deterministe, 1=equivalent DDPM)

        Returns:
            images: tensor (n_samples, 3, H, W) dans [-1, 1]
        """
        device = device or DEVICE

        # Sous-echantillonnage uniforme des timesteps
        timesteps = np.linspace(0, self.n_timesteps - 1, num_steps, dtype=int)[::-1]
        timesteps = list(timesteps)

        x = torch.randn(n_samples, 3, image_size, image_size, device=device)

        for i, t in enumerate(tqdm(timesteps, desc="DDIM sampling")):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            predicted_noise = self.unet(x, t_batch)

            # Timestep precedent (-1 si dernier pas)
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1

            x = self.scheduler.ddim_reverse_step(x, t, t_prev, predicted_noise, eta=eta)

        x = x.clamp(-1, 1)
        return x
