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

Sauvegarde Google Drive :
  Si save_dir est fourni, les checkpoints, samples generes et historique
  des pertes sont sauvegardes automatiquement dans Drive.
  L'entrainement reprend automatiquement depuis le dernier checkpoint.
"""

import os
import json
import glob
import copy
import torch
import torch.optim as optim
from torchvision.utils import save_image
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

    Usage basique (sauvegarde locale) :
        trainer = DiffusionTrainer()
        trainer.train(dataloader)

    Usage avec Google Drive (sauvegarde persistante + auto-resume) :
        trainer = DiffusionTrainer(save_dir="/content/drive/MyDrive/SatelliteGAN-Outputs/diffusion")
        trainer.train(dataloader)
    """

    def __init__(self, config=None, save_dir=None):
        """
        Args:
            config: dict de configuration (defaut: DIFFUSION de config.py)
            save_dir: chemin Drive pour sauvegarder checkpoints/samples/pertes.
                      Si None, sauvegarde dans outputs/ (local).
                      Si fourni, cree la structure :
                        save_dir/checkpoints/
                        save_dir/samples/
                        save_dir/losses/
        """
        cfg = config or DIFFUSION
        self.cfg = cfg
        self.save_dir = save_dir

        # Configurer les chemins de sauvegarde
        if save_dir:
            self.checkpoint_dir = os.path.join(save_dir, 'checkpoints')
            self.samples_dir = os.path.join(save_dir, 'samples')
            self.losses_dir = os.path.join(save_dir, 'losses')
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(self.samples_dir, exist_ok=True)
            os.makedirs(self.losses_dir, exist_ok=True)
        else:
            self.checkpoint_dir = DIFFUSION_CKPT_DIR
            self.samples_dir = None
            self.losses_dir = None

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

    def _find_latest_checkpoint(self):
        """
        Cherche le dernier checkpoint dans checkpoint_dir.
        Retourne le chemin du fichier ou None si aucun checkpoint.
        """
        pattern = os.path.join(self.checkpoint_dir, 'epoch_*.pth')
        checkpoints = glob.glob(pattern)
        if not checkpoints:
            return None

        def _epoch_num(path):
            basename = os.path.basename(path)
            try:
                return int(basename.replace('epoch_', '').replace('.pth', ''))
            except ValueError:
                return -1

        checkpoints.sort(key=_epoch_num)
        return checkpoints[-1]

    def train(self, dataloader, n_epochs=None, resume_from=None):
        """
        Boucle d'entrainement.

        Si save_dir a ete fourni a l'initialisation, les checkpoints,
        samples generes et historique des pertes sont sauvegardes
        automatiquement dans Drive.

        Si resume_from est fourni, reprend depuis ce checkpoint.
        Sinon, si save_dir contient des checkpoints, reprend
        automatiquement depuis le plus recent (auto-resume).

        Args:
            dataloader: DataLoader d'images (chaque batch = (images, labels) ou (images,))
            n_epochs: nombre d'epochs
            resume_from: chemin .pth pour reprendre l'entrainement.
                         Si None ET save_dir contient des checkpoints,
                         reprend automatiquement depuis le dernier.

        Returns:
            historique des pertes
        """
        n_epochs = n_epochs or self.cfg['n_epochs']
        start_epoch = 0

        # -- Auto-resume : chercher le dernier checkpoint --
        ckpt_path = resume_from
        if ckpt_path is None and self.save_dir:
            ckpt_path = self._find_latest_checkpoint()

        if ckpt_path and os.path.exists(ckpt_path):
            start_epoch = self.load_checkpoint(ckpt_path)
            print(f"Reprise de l'entrainement a l'epoch {start_epoch}/{n_epochs}")
        else:
            if self.save_dir:
                print(f"Sauvegarde activee : {self.save_dir}")
            print(f"Entrainement depuis le debut (epoch 1/{n_epochs})")

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
                self._save_samples(epoch + 1)
                self._save_loss_history()

        # Sauvegarde finale
        self._save_final_model()
        self._save_loss_history()

        return self.history

    def save_checkpoint(self, epoch):
        """Sauvegarde le modele, EMA, optimiseur et historique."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'ema': self.ema.shadow.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'history': self.history,
        }, path)
        print(f"Checkpoint sauvegarde : {path}")

    def load_checkpoint(self, path):
        """
        Charge un checkpoint pour reprendre l'entrainement ou l'inference.

        Restaure le modele, EMA, optimiseur et historique des pertes.
        Retourne le numero d'epoch (= epoch de depart pour reprendre).
        """
        ckpt = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(ckpt['model'])
        if 'ema' in ckpt:
            self.ema.shadow.load_state_dict(ckpt['ema'])
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'history' in ckpt:
            self.history = ckpt['history']
        try:
            epoch = ckpt['epoch']
        except (KeyError, TypeError):
            epoch = 0
        print(f"Checkpoint charge : epoch {epoch}")
        return epoch

    def _save_samples(self, epoch, n_samples=16):
        """Genere et sauvegarde des images exemples pendant l'entrainement."""
        if not self.samples_dir:
            return

        try:
            self.ema.shadow.eval()
            with torch.no_grad():
                samples = self.ema.shadow.sample_fast(n_samples, 64, DEVICE)

            path = os.path.join(self.samples_dir, f'epoch_{epoch}.png')
            save_image(samples, path, nrow=4, normalize=True)
            print(f"Samples sauvegardes : {path}")
        except Exception as e:
            print(f"Avertissement : sauvegarde samples echouee ({e})")

    def _save_loss_history(self):
        """Sauvegarde l'historique des pertes en JSON."""
        if not self.losses_dir:
            return
        path = os.path.join(self.losses_dir, 'loss_history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def _save_final_model(self):
        """Sauvegarde le modele final (poids + EMA, sans optimiseur)."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, 'final.pth')
        torch.save({
            'model': self.model.state_dict(),
            'ema': self.ema.shadow.state_dict(),
        }, path)
        print(f"Modele final sauvegarde : {path}")

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
