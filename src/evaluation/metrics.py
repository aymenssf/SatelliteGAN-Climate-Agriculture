"""
metrics.py
----------
Metriques d'evaluation pour les images generees.

Trois metriques principales :
  - SSIM (Structural Similarity Index) : mesure la similarite structurelle
    entre deux images. Valeurs dans [-1, 1], 1 = identique.
  - PSNR (Peak Signal-to-Noise Ratio) : mesure le rapport signal/bruit.
    Plus c'est haut, meilleur c'est. Typiquement 20-40 dB.
  - FID (Frechet Inception Distance) : mesure la distance entre les
    distributions d'images reelles et generees. Plus c'est bas, mieux c'est.

SSIM et PSNR mesurent la fidelite image-par-image.
FID mesure la qualite globale de la distribution d'images generees.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg


def compute_ssim(img1, img2, window_size=11, data_range=2.0):
    """
    Calcule le SSIM entre deux images ou batches d'images.

    Le SSIM considere trois composantes :
      - Luminance : difference de moyenne
      - Contraste : difference de variance
      - Structure : correlation des pixels

    Args:
        img1, img2: tensors (B, C, H, W) dans [-1, 1]
        window_size: taille du filtre gaussien local
        data_range: plage des valeurs (2.0 pour [-1, 1])

    Returns:
        score SSIM moyen (scalaire)
    """
    # Constantes de stabilite (cf. papier SSIM original)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Filtre gaussien pour le calcul local
    channel = img1.shape[1]
    kernel = _gaussian_kernel(window_size, 1.5).to(img1.device)
    kernel = kernel.expand(channel, 1, window_size, window_size)

    # Moyennes locales
    mu1 = F.conv2d(img1, kernel, groups=channel, padding=window_size // 2)
    mu2 = F.conv2d(img2, kernel, groups=channel, padding=window_size // 2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    # Variances et covariance locales
    sigma1_sq = F.conv2d(img1 ** 2, kernel, groups=channel, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, kernel, groups=channel, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, groups=channel, padding=window_size // 2) - mu1_mu2

    # Formule SSIM
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator

    return ssim_map.mean().item()


def _gaussian_kernel(size, sigma):
    """Cree un noyau gaussien 2D normalise."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = torch.outer(g, g)
    g = g / g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def compute_psnr(img1, img2, data_range=2.0):
    """
    Calcule le PSNR entre deux images.

    PSNR = 10 * log10(MAX^2 / MSE)

    Un PSNR eleve signifie que les images sont proches pixel-a-pixel.
    Typiquement :
      - > 30 dB : bonne qualite
      - > 40 dB : excellente qualite

    Args:
        img1, img2: tensors (B, C, H, W)
        data_range: plage des valeurs (2.0 pour [-1, 1])

    Returns:
        score PSNR moyen en dB (scalaire)
    """
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(data_range ** 2 / mse)


def compute_fid(real_features, generated_features):
    """
    Calcule la Frechet Inception Distance (FID) entre deux ensembles de features.

    FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2 * sqrt(Sigma_r * Sigma_g))

    Mesure la distance entre les distributions gaussiennes ajustees
    aux features d'un reseau pre-entraine (InceptionV3).

    Plus le FID est bas, plus les images generees ressemblent aux vraies.
    Un FID de 0 signifie des distributions identiques.

    Args:
        real_features: array (N, D) de features pour les images reelles
        generated_features: array (N, D) de features pour les images generees

    Returns:
        score FID (scalaire, plus bas = mieux)
    """
    # Convertir en numpy si necessaire
    if isinstance(real_features, torch.Tensor):
        real_features = real_features.cpu().numpy()
    if isinstance(generated_features, torch.Tensor):
        generated_features = generated_features.cpu().numpy()

    # Statistiques des distributions
    mu_r = np.mean(real_features, axis=0)
    mu_g = np.mean(generated_features, axis=0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_g = np.cov(generated_features, rowvar=False)

    # Regularisation pour eviter les matrices singulieres
    # (necessaire quand N < D, standard dans les implementations FID)
    eps = 1e-6
    sigma_r += eps * np.eye(sigma_r.shape[0])
    sigma_g += eps * np.eye(sigma_g.shape[0])

    # Distance entre les moyennes
    diff = mu_r - mu_g
    mean_diff = diff @ diff

    # Racine carree du produit des covariances
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)

    # Gestion des valeurs imaginaires dues a l'imprecision numerique
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # FID
    fid = mean_diff + np.trace(sigma_r + sigma_g - 2 * covmean)
    return float(fid)


class InceptionFeatureExtractor(nn.Module):
    """
    Extracteur de features InceptionV3 pre-entraine pour le calcul du FID.

    Utilise InceptionV3 (Szegedy et al., 2016) pre-entraine sur ImageNet,
    le standard pour le calcul du FID (Heusel et al., 2017).
    Les images sont redimensionnees a 299x299 et normalisees selon ImageNet.
    La couche de classification est remplacee par Identity pour obtenir
    le vecteur de features 2048-D du pool layer.
    """

    def __init__(self):
        super().__init__()
        from torchvision import models
        inception = models.inception_v3(weights='DEFAULT')
        inception.fc = nn.Identity()
        inception.eval()
        self.model = inception
        # Normalisation ImageNet
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        # [-1, 1] -> [0, 1] -> normalisation ImageNet
        x = (x + 1) / 2
        x = (x - self.mean) / self.std
        # Resize 299x299 (requis par InceptionV3)
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        return self.model(x)


# Alias pour compatibilite avec le code existant
SimpleFeatureExtractor = InceptionFeatureExtractor


@torch.no_grad()
def extract_features(images, extractor, batch_size=8, device=None):
    """
    Extrait les features d'un ensemble d'images.

    Args:
        images: tensor (N, C, H, W) ou liste de tensors
        extractor: reseau extracteur de features
        batch_size: taille des mini-batches
        device: device

    Returns:
        features: tensor (N, D)
    """
    from src.config import DEVICE as DEFAULT_DEVICE
    device = device or DEFAULT_DEVICE
    extractor = extractor.to(device).eval()

    all_features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device)
        feats = extractor(batch)
        all_features.append(feats.cpu())

    return torch.cat(all_features, dim=0)
