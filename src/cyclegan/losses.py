"""
losses.py
---------
Fonctions de perte pour le CycleGAN.

Le CycleGAN utilise trois types de pertes :
  1. Adversariale (LSGAN) : le discriminateur doit distinguer vrai/faux
  2. Cycle-consistency : G(F(x)) ≈ x et F(G(y)) ≈ y
  3. Identity : G(y) ≈ y et F(x) ≈ x (regularisation optionnelle)

On utilise la perte des moindres carres (LSGAN, Mao et al. 2017)
plutot que la BCE classique car elle stabilise l'entrainement et
evite la saturation des gradients.
"""

import torch
import torch.nn as nn


class CycleGANLoss(nn.Module):
    """
    Regroupe toutes les pertes du CycleGAN dans un seul module.

    Usage :
        loss_fn = CycleGANLoss(lambda_cycle=10.0, lambda_identity=5.0)
        loss_G, loss_D = loss_fn(...)
    """

    def __init__(self, lambda_cycle=10.0, lambda_identity=5.0):
        super().__init__()
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

        # Perte adversariale : MSE (Least Squares GAN)
        self.adversarial = nn.MSELoss()

        # Perte de reconstruction cyclique : L1 (preserve les details)
        self.cycle = nn.L1Loss()

        # Perte d'identite : L1
        self.identity = nn.L1Loss()

    def adversarial_loss(self, pred, is_real):
        """
        Perte adversariale LSGAN.

        Pour le discriminateur :
          - real -> target = 1
          - fake -> target = 0

        Pour le generateur :
          - fake -> target = 1 (on veut tromper le discriminateur)
        """
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.adversarial(pred, target)

    def cycle_consistency_loss(self, reconstructed, original):
        """
        Perte de coherence cyclique.

        Si on transforme une image A->B puis B->A, on doit retrouver A.
        |F(G(x)) - x| + |G(F(y)) - y|

        L1 plutot que L2 car L1 preserve mieux les details
        (L2 a tendance a produire des images floues).
        """
        return self.cycle(reconstructed, original) * self.lambda_cycle

    def identity_loss(self, same, original):
        """
        Perte d'identite (regularisation).

        Si on donne au generateur G une image deja dans son domaine cible,
        il devrait la laisser inchangee : G(y) ≈ y.

        Aide a preserver les teintes de couleur et evite des transformations
        trop agressives.
        """
        return self.identity(same, original) * self.lambda_identity

    def generator_loss(self, fake_pred_a, fake_pred_b,
                       cycle_a, real_a, cycle_b, real_b,
                       idt_a=None, idt_b=None):
        """
        Perte totale du generateur.

        Args:
            fake_pred_a : prediction du discriminateur A sur les fausses images A
            fake_pred_b : prediction du discriminateur B sur les fausses images B
            cycle_a : images A reconstruites apres le cycle A->B->A
            real_a : vraies images A
            cycle_b : images B reconstruites apres le cycle B->A->B
            real_b : vraies images B
            idt_a : G_B2A(real_a), si identity loss activee
            idt_b : G_A2B(real_b), si identity loss activee

        Returns:
            loss_total, dict de composantes
        """
        # Pertes adversariales : on veut que le discriminateur dise "vrai"
        loss_adv_a = self.adversarial_loss(fake_pred_a, is_real=True)
        loss_adv_b = self.adversarial_loss(fake_pred_b, is_real=True)

        # Pertes cycliques
        loss_cycle_a = self.cycle_consistency_loss(cycle_a, real_a)
        loss_cycle_b = self.cycle_consistency_loss(cycle_b, real_b)

        total = loss_adv_a + loss_adv_b + loss_cycle_a + loss_cycle_b

        # Perte d'identite (optionnelle)
        loss_idt = torch.tensor(0.0)
        if idt_a is not None and idt_b is not None and self.lambda_identity > 0:
            loss_idt_a = self.identity_loss(idt_a, real_a)
            loss_idt_b = self.identity_loss(idt_b, real_b)
            loss_idt = loss_idt_a + loss_idt_b
            total = total + loss_idt

        components = {
            'adv_a': loss_adv_a.item(),
            'adv_b': loss_adv_b.item(),
            'cycle_a': loss_cycle_a.item(),
            'cycle_b': loss_cycle_b.item(),
            'identity': loss_idt.item(),
        }

        return total, components

    def discriminator_loss(self, real_pred, fake_pred):
        """
        Perte du discriminateur.

        Le discriminateur doit :
          - Predire 1 (reel) pour les vraies images
          - Predire 0 (faux) pour les images generees

        On divise par 2 pour ralentir le discriminateur par rapport
        au generateur (convention standard).
        """
        loss_real = self.adversarial_loss(real_pred, is_real=True)
        loss_fake = self.adversarial_loss(fake_pred, is_real=False)
        return (loss_real + loss_fake) * 0.5
