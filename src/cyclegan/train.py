"""
train.py
--------
Boucle d'entrainement du CycleGAN.

Le CycleGAN s'entraine avec deux generateurs et deux discriminateurs :
  - G_A2B : transforme domaine A (normal) -> domaine B (secheresse)
  - G_B2A : transforme domaine B (secheresse) -> domaine A (normal)
  - D_A : discrimine les vraies images A des fausses
  - D_B : discrimine les vraies images B des fausses

A chaque iteration :
  1. Generer les fausses images et les reconstructions cycliques
  2. Mettre a jour les generateurs (pertes adversariales + cycliques + identite)
  3. Mettre a jour les discriminateurs (pertes adversariales avec replay buffer)
"""

import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR as TorchLambdaLR
from tqdm import tqdm

from src.config import DEVICE, CYCLEGAN, CYCLEGAN_CKPT_DIR
from src.cyclegan.models import Generator, Discriminator
from src.cyclegan.losses import CycleGANLoss
from src.cyclegan.utils import ReplayBuffer, init_weights, LambdaLR


class CycleGANTrainer:
    """
    Classe qui encapsule toute la logique d'entrainement du CycleGAN.

    Usage :
        trainer = CycleGANTrainer()
        trainer.train(dataloader, n_epochs=100)
    """

    def __init__(self, config=None):
        """
        Initialise les modeles, optimiseurs, et pertes.

        Args:
            config: dict de configuration (defaut: CYCLEGAN de config.py)
        """
        cfg = config or CYCLEGAN
        self.cfg = cfg

        # Generateurs
        self.G_A2B = Generator(
            n_residual_blocks=cfg['n_residual_blocks']
        ).to(DEVICE)
        self.G_B2A = Generator(
            n_residual_blocks=cfg['n_residual_blocks']
        ).to(DEVICE)

        # Discriminateurs
        self.D_A = Discriminator().to(DEVICE)
        self.D_B = Discriminator().to(DEVICE)

        # Initialisation des poids
        self.G_A2B.apply(init_weights)
        self.G_B2A.apply(init_weights)
        self.D_A.apply(init_weights)
        self.D_B.apply(init_weights)

        # Pertes
        self.loss_fn = CycleGANLoss(
            lambda_cycle=cfg['lambda_cycle'],
            lambda_identity=cfg['lambda_identity'],
        )

        # Optimiseurs : un pour les deux generateurs, un par discriminateur
        self.opt_G = optim.Adam(
            list(self.G_A2B.parameters()) + list(self.G_B2A.parameters()),
            lr=cfg['lr'],
            betas=cfg['betas'],
        )
        self.opt_D_A = optim.Adam(
            self.D_A.parameters(),
            lr=cfg['lr'],
            betas=cfg['betas'],
        )
        self.opt_D_B = optim.Adam(
            self.D_B.parameters(),
            lr=cfg['lr'],
            betas=cfg['betas'],
        )

        # Schedulers de learning rate : decay lineaire apres decay_epoch
        lr_lambda = LambdaLR(cfg['n_epochs'], cfg['decay_epoch'])
        self.sched_G = TorchLambdaLR(self.opt_G, lr_lambda.step)
        self.sched_D_A = TorchLambdaLR(self.opt_D_A, lr_lambda.step)
        self.sched_D_B = TorchLambdaLR(self.opt_D_B, lr_lambda.step)

        # Buffers de replay pour les discriminateurs
        self.buffer_A = ReplayBuffer(cfg['replay_buffer_size'])
        self.buffer_B = ReplayBuffer(cfg['replay_buffer_size'])

        # Historique des pertes pour le suivi
        self.history = {
            'loss_G': [], 'loss_D_A': [], 'loss_D_B': [],
            'loss_cycle': [], 'loss_identity': [],
        }

    def train_step(self, real_a, real_b):
        """
        Execute une iteration d'entrainement.

        Returns:
            dict avec les composantes des pertes
        """
        real_a = real_a.to(DEVICE)
        real_b = real_b.to(DEVICE)

        # ---- Generateurs ----
        self.opt_G.zero_grad()

        # Generation
        fake_b = self.G_A2B(real_a)           # A -> B
        fake_a = self.G_B2A(real_b)           # B -> A
        cycle_a = self.G_B2A(fake_b)          # A -> B -> A
        cycle_b = self.G_A2B(fake_a)          # B -> A -> B

        # Identity (optionnel mais aide a preserver les couleurs)
        idt_a = self.G_B2A(real_a)            # B2A applique sur A -> devrait rien changer
        idt_b = self.G_A2B(real_b)            # A2B applique sur B -> devrait rien changer

        # Predictions des discriminateurs sur les fausses images
        pred_fake_a = self.D_A(fake_a)
        pred_fake_b = self.D_B(fake_b)

        # Perte generateur
        loss_G, components = self.loss_fn.generator_loss(
            pred_fake_a, pred_fake_b,
            cycle_a, real_a, cycle_b, real_b,
            idt_a, idt_b,
        )

        loss_G.backward()
        self.opt_G.step()

        # ---- Discriminateur A ----
        self.opt_D_A.zero_grad()

        # Utiliser le replay buffer pour les fausses images
        fake_a_buffer = self.buffer_A.push_and_pop(fake_a.detach())
        pred_real_a = self.D_A(real_a)
        pred_fake_a = self.D_A(fake_a_buffer)

        loss_D_A = self.loss_fn.discriminator_loss(pred_real_a, pred_fake_a)
        loss_D_A.backward()
        self.opt_D_A.step()

        # ---- Discriminateur B ----
        self.opt_D_B.zero_grad()

        fake_b_buffer = self.buffer_B.push_and_pop(fake_b.detach())
        pred_real_b = self.D_B(real_b)
        pred_fake_b = self.D_B(fake_b_buffer)

        loss_D_B = self.loss_fn.discriminator_loss(pred_real_b, pred_fake_b)
        loss_D_B.backward()
        self.opt_D_B.step()

        return {
            'loss_G': loss_G.item(),
            'loss_D_A': loss_D_A.item(),
            'loss_D_B': loss_D_B.item(),
            'loss_cycle': components['cycle_a'] + components['cycle_b'],
            'loss_identity': components['identity'],
        }

    def train(self, dataloader, n_epochs=None, start_epoch=0):
        """
        Boucle d'entrainement complete.

        Args:
            dataloader: DataLoader retournant (images_A, images_B)
            n_epochs: nombre d'epochs (defaut: cfg)
            start_epoch: epoch de depart (pour reprendre un entrainement)

        Returns:
            dict historique des pertes
        """
        n_epochs = n_epochs or self.cfg['n_epochs']

        for epoch in range(start_epoch, n_epochs):
            epoch_losses = {k: 0.0 for k in self.history}
            n_batches = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
            for real_a, real_b in pbar:
                losses = self.train_step(real_a, real_b)
                n_batches += 1

                for k, v in losses.items():
                    epoch_losses[k] += v

                # Affichage dans la barre de progression
                pbar.set_postfix({
                    'G': f"{losses['loss_G']:.3f}",
                    'D_A': f"{losses['loss_D_A']:.3f}",
                    'D_B': f"{losses['loss_D_B']:.3f}",
                })

            # Moyenne par epoch
            for k in epoch_losses:
                avg = epoch_losses[k] / max(n_batches, 1)
                self.history[k].append(avg)

            # Mise a jour du learning rate
            self.sched_G.step()
            self.sched_D_A.step()
            self.sched_D_B.step()

            # Log
            print(f"[Epoch {epoch+1}] "
                  f"G: {self.history['loss_G'][-1]:.4f} | "
                  f"D_A: {self.history['loss_D_A'][-1]:.4f} | "
                  f"D_B: {self.history['loss_D_B'][-1]:.4f} | "
                  f"Cycle: {self.history['loss_cycle'][-1]:.4f}")

            # Sauvegarde periodique
            if (epoch + 1) % self.cfg['save_every'] == 0:
                self.save_checkpoint(epoch + 1)

        return self.history

    def save_checkpoint(self, epoch):
        """Sauvegarde les modeles et optimiseurs."""
        os.makedirs(CYCLEGAN_CKPT_DIR, exist_ok=True)
        path = os.path.join(CYCLEGAN_CKPT_DIR, f'cyclegan_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'G_A2B': self.G_A2B.state_dict(),
            'G_B2A': self.G_B2A.state_dict(),
            'D_A': self.D_A.state_dict(),
            'D_B': self.D_B.state_dict(),
            'opt_G': self.opt_G.state_dict(),
            'opt_D_A': self.opt_D_A.state_dict(),
            'opt_D_B': self.opt_D_B.state_dict(),
        }, path)
        print(f"Checkpoint sauvegarde : {path}")

    def load_checkpoint(self, path):
        """Charge un checkpoint pour reprendre l'entrainement ou l'inference."""
        ckpt = torch.load(path, map_location=DEVICE)
        self.G_A2B.load_state_dict(ckpt['G_A2B'])
        self.G_B2A.load_state_dict(ckpt['G_B2A'])
        self.D_A.load_state_dict(ckpt['D_A'])
        self.D_B.load_state_dict(ckpt['D_B'])
        if 'opt_G' in ckpt:
            self.opt_G.load_state_dict(ckpt['opt_G'])
            self.opt_D_A.load_state_dict(ckpt['opt_D_A'])
            self.opt_D_B.load_state_dict(ckpt['opt_D_B'])
        print(f"Checkpoint charge : epoch {ckpt['epoch']}")
        return ckpt['epoch']

    @torch.no_grad()
    def generate(self, images, direction='A2B'):
        """
        Genere des images avec le generateur entraine.

        Args:
            images: tensor (B, C, H, W)
            direction: 'A2B' (normal -> secheresse) ou 'B2A' (secheresse -> normal)

        Returns:
            tensor (B, C, H, W) d'images generees
        """
        images = images.to(DEVICE)
        if direction == 'A2B':
            return self.G_A2B(images)
        else:
            return self.G_B2A(images)
