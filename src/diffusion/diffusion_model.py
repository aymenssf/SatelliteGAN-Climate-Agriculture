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
        Sampling accelere avec moins de pas.

        Au lieu de faire les 1000 pas, on en fait seulement
        sampling_steps (ex: 50) en sautant des pas regulierement.
        Qualite legerement inferieure mais beaucoup plus rapide.

        Args:
            n_samples: nombre d'images a generer
            image_size: taille spatiale des images
            device: device

        Returns:
            images: tensor (n_samples, 3, H, W) dans [-1, 1]
        """
        device = device or DEVICE

        # Sous-echantillonner les timesteps
        step_size = self.n_timesteps // self.sampling_steps
        timesteps = list(range(0, self.n_timesteps, step_size))[::-1]

        x = torch.randn(n_samples, 3, image_size, image_size, device=device)

        for t in tqdm(timesteps, desc="Fast sampling"):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            predicted_noise = self.unet(x, t_batch)
            x = self.scheduler.reverse_step(x, t, predicted_noise)

        x = x.clamp(-1, 1)
        return x
