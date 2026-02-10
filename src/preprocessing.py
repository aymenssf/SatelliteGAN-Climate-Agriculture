"""
preprocessing.py
----------------
Transformations et augmentation des images EuroSAT.

Contient :
  - Les transformations standard pour l'entrainement (augmentation + normalisation)
  - La simulation de secheresse par modification spectrale
  - Les fonctions de denormalisation pour la visualisation
"""

import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageEnhance

from src.config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD


# ------------------------------------------------------------------
# Transformations standard
# ------------------------------------------------------------------

def get_train_transform():
    """
    Transformations pour l'entrainement.
    Augmentations legeres : flip horizontal + petite rotation.
    On evite les augmentations trop agressives qui pourraient
    denaturer la signification spectrale des images satellite.
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])


def get_eval_transform():
    """Transformations pour l'evaluation (pas d'augmentation)."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])


def get_cyclegan_transform():
    """
    Transformations pour le CycleGAN.
    Similaire a l'entrainement, avec un random crop en plus
    pour introduire de la variabilite spatiale.
    """
    return transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 1.1)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])


# ------------------------------------------------------------------
# Denormalisation (pour visualiser les images)
# ------------------------------------------------------------------

def denormalize(tensor, mean=NORMALIZE_MEAN, std=NORMALIZE_STD):
    """
    Inverse la normalisation pour afficher une image.
    Input : tensor normalise dans [-1, 1]
    Output : tensor dans [0, 1] (compatible matplotlib)
    """
    # Clone pour ne pas modifier l'original
    result = tensor.clone()
    for t, m, s in zip(result, mean, std):
        t.mul_(s).add_(m)
    return result.clamp(0, 1)


def tensor_to_numpy(tensor):
    """Convertit un tensor PyTorch (C, H, W) en array numpy (H, W, C)."""
    if tensor.dim() == 4:
        tensor = tensor[0]  # prend le premier element du batch
    img = denormalize(tensor)
    return img.permute(1, 2, 0).cpu().numpy()


# ------------------------------------------------------------------
# Simulation de secheresse
# ------------------------------------------------------------------

def simulate_drought(image, severity=0.6):
    """
    Simule un effet de secheresse sur une image satellite.

    L'idee est de reproduire visuellement ce qu'on observe sur les images
    satellite en periode de secheresse :
      - Reduction de la composante verte (vegetation stressée -> moins de chlorophylle)
      - Augmentation de la composante rouge/jaune (sol nu, vegetation seche)
      - Reduction de la saturation globale
      - Leggere augmentation de la luminosite (sol sec = plus reflechissant)

    Cette simulation est grossiere -- c'est justement pour ca qu'on utilise
    un CycleGAN pour apprendre une transformation plus realiste.

    Args:
        image: PIL Image RGB
        severity: entre 0 (pas de changement) et 1 (secheresse severe)

    Returns:
        PIL Image modifiee
    """
    img_array = np.array(image, dtype=np.float32)

    # Reduction du vert (canal 1 en RGB)
    # En secheresse, la chlorophylle diminue => le vert recule
    green_reduction = 1.0 - (severity * 0.35)
    img_array[:, :, 1] *= green_reduction

    # Augmentation du rouge (canal 0)
    # La vegetation seche reflechit plus dans le rouge
    red_boost = 1.0 + (severity * 0.2)
    img_array[:, :, 0] *= red_boost

    # Ajout d'une teinte jaune-brune au sol
    # On melange un peu de rouge dans le vert quand la vegetation meurt
    yellow_shift = severity * 0.1
    img_array[:, :, 1] += img_array[:, :, 0] * yellow_shift

    # Clamp dans [0, 255]
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    drought_img = Image.fromarray(img_array)

    # Reduction de la saturation (image plus "delavee")
    enhancer = ImageEnhance.Color(drought_img)
    drought_img = enhancer.enhance(1.0 - severity * 0.3)

    # Legere augmentation de la luminosite
    enhancer = ImageEnhance.Brightness(drought_img)
    drought_img = enhancer.enhance(1.0 + severity * 0.1)

    return drought_img


def compute_ndvi_proxy(image_tensor):
    """
    Calcule un proxy du NDVI a partir des bandes RGB.

    Le vrai NDVI utilise les bandes NIR et Rouge de Sentinel-2 :
        NDVI = (NIR - Rouge) / (NIR + Rouge)

    Comme on travaille en RGB (3 bandes), on utilise un proxy :
        NDVI_proxy = (Vert - Rouge) / (Vert + Rouge + epsilon)

    C'est une approximation grossiere, mais elle capture quand meme
    la difference entre vegetation saine (beaucoup de vert) et
    vegetation stressée (plus de rouge). Suffisant pour notre analyse.

    Args:
        image_tensor: tensor (C, H, W) normalise [-1, 1]

    Returns:
        ndvi_map: tensor (H, W) avec valeurs dans [-1, 1]
    """
    # Denormaliser d'abord
    img = denormalize(image_tensor)
    red = img[0]    # canal R
    green = img[1]  # canal G

    eps = 1e-7
    ndvi = (green - red) / (green + red + eps)
    return ndvi
