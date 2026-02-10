"""
train.py
--------
Boucle d'entrainement du modele de diffusion (DDPM).

L'entrainement est plus simple que pour le CycleGAN :
  - A chaque iteration, on echantillonne un bruit et un timestep
  - Le modele predit le bruit
  - La perte est le MSE entre bruit reel et predit
  - Pas de discriminateur, pas de pertes multiples

On utilise EMA (Exponential Moving Average) des poids pour
ameliorer la qualite du sampling (convention standard en diffusion).
"""

import os
import copy
import torch
import torch.optim as optim
from tqdm import tqdm

from src.config import DEVICE, DIFFUSION, DIFFUSION_CKPT_DIR
from src.diffusion.diffusion_model import DDPM


class EMA:
    """
    Exponential Moving Average des poids du modele.

    Maintient une copie "lissee" des poids : a chaque update,
    les poids EMA sont mis a jour comme :
        ema_weight = decay * ema_weight + (1 - decay) * new_weight

    Cela produit des images avec moins d'artefacts au sampling.
    Decay typique : 0.9999.
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()

        # Desactiver les gradients pour le modele EMA
        for param in self.shadow.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        """Met a jour les poids EMA."""
        for ema_param, model_param in zip(
            self.shadow.parameters(), model.parameters()
        ):
            ema_param.data.mul_(self.decay).add_(
                model_param.data, alpha=1.0 - self.decay
            )

    def forward(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)


class DiffusionTrainer:
    """
    Classe d'entrainement pour le DDPM.

    Usage :
        trainer = DiffusionTrainer()
        trainer.train(dataloader)
    """

    def __init__(self, config=None):
        cfg = config or DIFFUSION
        self.cfg = cfg

        # Modele DDPM
        self.model = DDPM(config=cfg).to(DEVICE)

        # EMA
        self.ema = EMA(self.model)

        # Optimiseur : Adam avec lr standard
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg['lr'],
        )

        # Historique
        self.history = {'loss': []}

    def train(self, dataloader, n_epochs=None, start_epoch=0):
        """
        Boucle d'entrainement.

        Args:
            dataloader: DataLoader d'images (chaque batch = (images, labels))
            n_epochs: nombre d'epochs
            start_epoch: epoch de depart

        Returns:
            historique des pertes
        """
        n_epochs = n_epochs or self.cfg['n_epochs']
        self.model.train()

        for epoch in range(start_epoch, n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
            for batch in pbar:
                # Le dataloader peut retourner (images, labels) ou juste images
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(DEVICE)

                # Forward + backward
                self.optimizer.zero_grad()
                loss = self.model(images)
                loss.backward()

                # Gradient clipping pour la stabilite
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()

                # Mise a jour EMA
                self.ema.update(self.model)

                epoch_loss += loss.item()
                n_batches += 1
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_loss = epoch_loss / max(n_batches, 1)
            self.history['loss'].append(avg_loss)
            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

            # Sauvegarde periodique
            if (epoch + 1) % self.cfg['save_every'] == 0:
                self.save_checkpoint(epoch + 1)

        return self.history

    def save_checkpoint(self, epoch):
        """Sauvegarde le modele et l'optimiseur."""
        os.makedirs(DIFFUSION_CKPT_DIR, exist_ok=True)
        path = os.path.join(DIFFUSION_CKPT_DIR, f'ddpm_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'ema': self.ema.shadow.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'history': self.history,
        }, path)
        print(f"Checkpoint sauvegarde : {path}")

    def load_checkpoint(self, path):
        """Charge un checkpoint."""
        ckpt = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(ckpt['model'])
        if 'ema' in ckpt:
            self.ema.shadow.load_state_dict(ckpt['ema'])
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'history' in ckpt:
            self.history = ckpt['history']
        print(f"Checkpoint charge : epoch {ckpt['epoch']}")
        return ckpt['epoch']

    @torch.no_grad()
    def generate(self, n_samples=16, image_size=64, use_ema=True, fast=True):
        """
        Genere des images avec le modele entraine.

        Args:
            n_samples: nombre d'images a generer
            image_size: taille des images
            use_ema: utiliser les poids EMA (recommande)
            fast: utiliser le sampling accelere

        Returns:
            tensor (n_samples, 3, H, W) dans [-1, 1]
        """
        model = self.ema.shadow if use_ema else self.model
        model.eval()

        if fast:
            return model.sample_fast(n_samples, image_size, DEVICE)
        else:
            return model.sample(n_samples, image_size, DEVICE)
