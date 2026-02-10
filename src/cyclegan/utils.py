"""
utils.py
--------
Utilitaires pour l'entrainement du CycleGAN.

Contient le ReplayBuffer, une technique classique pour stabiliser
l'entrainement des GANs : au lieu de montrer uniquement les dernières
images generees au discriminateur, on maintient un buffer d'images
passees et on en pioche aleatoirement.

Reference : Shrivastava et al. 2017, "Learning from Simulated and
Unsupervised Images through Adversarial Training"
"""

import torch
import random


class ReplayBuffer:
    """
    Buffer de replay pour le discriminateur.

    Stocke les N dernieres images generees. Quand le buffer est plein,
    on remplace aleatoirement une ancienne image (50% de chance) ou
    on retourne directement la nouvelle image.

    Cela empeche le discriminateur de trop s'adapter aux dernieres
    generations du generateur, ce qui stabilise l'entrainement.
    """

    def __init__(self, max_size=50):
        self.max_size = max_size
        self.buffer = []

    def push_and_pop(self, images):
        """
        Ajoute des images au buffer et retourne un batch (mix ancien/nouveau).

        Args:
            images: tensor (B, C, H, W) d'images generees

        Returns:
            tensor (B, C, H, W) d'images (certaines du buffer, d'autres nouvelles)
        """
        result = []
        for img in images:
            img = img.unsqueeze(0)  # (1, C, H, W)

            if len(self.buffer) < self.max_size:
                # Buffer pas encore plein : on stocke et retourne l'image
                self.buffer.append(img.clone())
                result.append(img)
            else:
                # Buffer plein : 50% chance de piocher dans le buffer
                if random.random() > 0.5:
                    # Piocher une ancienne image et la remplacer
                    idx = random.randint(0, self.max_size - 1)
                    old_img = self.buffer[idx].clone()
                    self.buffer[idx] = img.clone()
                    result.append(old_img)
                else:
                    # Retourner la nouvelle image directement
                    result.append(img)

        return torch.cat(result, dim=0)


def init_weights(module, gain=0.02):
    """
    Initialisation des poids du reseau.

    Utilise l'initialisation normale (mean=0, std=gain) comme dans
    le papier CycleGAN original. L'InstanceNorm est initialisee
    avec weight=1 et bias=0.
    """
    classname = module.__class__.__name__
    if hasattr(module, 'weight') and 'Conv' in classname:
        torch.nn.init.normal_(module.weight.data, 0.0, gain)
    elif 'InstanceNorm' in classname and module.weight is not None:
        torch.nn.init.normal_(module.weight.data, 1.0, gain)
        torch.nn.init.constant_(module.bias.data, 0.0)


class LambdaLR:
    """
    Planning du learning rate avec decay lineaire.

    Le lr reste constant pendant decay_epoch epochs, puis decroit
    lineairement jusqu'a 0 sur les epochs restants.
    C'est le schedule standard du papier CycleGAN.
    """

    def __init__(self, n_epochs, decay_epoch):
        self.n_epochs = n_epochs
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        """Retourne le facteur multiplicatif pour le lr."""
        return 1.0 - max(0, epoch - self.decay_epoch) / (self.n_epochs - self.decay_epoch)
