"""
models.py
---------
Architectures du CycleGAN : Generateur (ResNet) et Discriminateur (PatchGAN).

Le generateur suit l'architecture du papier original de Zhu et al. (2017) :
  - Encodeur (convolutions descendantes)
  - Blocs residuels (transformation dans l'espace des features)
  - Decodeur (convolutions montantes)

Le discriminateur est un PatchGAN 70x70 : il classifie chaque patch
de 70x70 pixels comme reel ou faux, plutot que l'image entiere.
Cela force le generateur a produire des textures realistes localement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Blocs de base
# ------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Bloc residuel avec deux convolutions 3x3 et une connexion residuelle.
    Instance Normalization est preferee a Batch Normalization pour le style transfer
    (cf. papier "Instance Normalization: The Missing Ingredient for Fast Stylization").
    """

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


# ------------------------------------------------------------------
# Generateur
# ------------------------------------------------------------------

class Generator(nn.Module):
    """
    Generateur ResNet pour CycleGAN.

    Architecture :
      1. Convolution initiale 7x7 (extraction de features)
      2. 2 convolutions descendantes (stride 2, doublement des canaux)
      3. N blocs residuels (transformation dans l'espace latent)
      4. 2 convolutions montantes (stride 1/2, reduction des canaux)
      5. Convolution finale 7x7 + tanh (sortie dans [-1, 1])

    On utilise ReflectionPad au lieu de ZeroPad pour eviter les artefacts
    de bord, surtout visibles sur les images satellites.
    """

    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=6, n_features=64):
        super().__init__()

        # Couche initiale : conv 7x7 pour capturer le contexte large
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, n_features, kernel_size=7, bias=False),
            nn.InstanceNorm2d(n_features),
            nn.ReLU(inplace=True),
        ]

        # Encodeur : 2 convolutions descendantes
        # 64 -> 128 -> 256 canaux ; resolution / 4
        in_f = n_features
        out_f = in_f * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_f, out_f, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_f),
                nn.ReLU(inplace=True),
            ]
            in_f = out_f
            out_f = in_f * 2

        # Blocs residuels : transformation dans l'espace des features
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_f)]

        # Decodeur : 2 convolutions montantes (transposees)
        # 256 -> 128 -> 64 canaux ; resolution * 4
        out_f = in_f // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(
                    in_f, out_f, kernel_size=3, stride=2,
                    padding=1, output_padding=1, bias=False
                ),
                nn.InstanceNorm2d(out_f),
                nn.ReLU(inplace=True),
            ]
            in_f = out_f
            out_f = in_f // 2

        # Couche finale : conv 7x7 -> 3 canaux RGB
        # tanh pour borner la sortie dans [-1, 1] (meme range que les images normalisees)
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_features, out_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ------------------------------------------------------------------
# Discriminateur
# ------------------------------------------------------------------

class Discriminator(nn.Module):
    """
    Discriminateur PatchGAN (Markovien).

    Au lieu de produire un seul scalaire reel/faux, il produit une carte
    de decision ou chaque element correspond a un patch de l'image source.
    Cela force le generateur a produire des details realistes a l'echelle locale.

    Architecture :
      - 4 couches de convolution avec stride 2 (sauf la derniere)
      - Pas d'Instance Norm sur la premiere couche (convention standard)
      - Sortie : feature map N x N (chaque valeur = score reel/faux pour un patch)

    Pour des images 64x64, la sortie est environ 6x6.
    """

    def __init__(self, in_channels=3, n_features=64):
        super().__init__()

        # Premiere couche : pas de normalisation (convention du papier pix2pix)
        layers = [
            nn.Conv2d(in_channels, n_features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Couches intermediaires : doublement progressif des canaux
        # 64 -> 128 -> 256 -> 512
        in_f = n_features
        for mult in [2, 4, 8]:
            out_f = min(n_features * mult, 512)
            layers += [
                nn.Conv2d(in_f, out_f, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_f),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_f = out_f

        # Couche finale : 1 canal de sortie (score reel/faux par patch)
        layers += [
            nn.Conv2d(in_f, 1, kernel_size=4, stride=1, padding=1),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
