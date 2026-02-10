"""
visualization.py
----------------
Fonctions de visualisation pour les resultats du projet.

Contient des utilitaires pour :
  - Afficher des grilles d'images (reelles, generees, comparaisons)
  - Visualiser les courbes de pertes pendant l'entrainement
  - Comparer cote-a-cote les images normales et leurs versions secheresse
  - Afficher les cartes NDVI
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.preprocessing import denormalize, tensor_to_numpy, compute_ndvi_proxy


def show_image_grid(images, n_cols=4, title=None, figsize=None, save_path=None):
    """
    Affiche une grille d'images.

    Args:
        images: tensor (N, C, H, W) normalise [-1, 1] ou liste de tensors
        n_cols: nombre de colonnes
        title: titre de la figure
        figsize: taille de la figure (auto si None)
        save_path: chemin pour sauvegarder (optionnel)
    """
    if isinstance(images, torch.Tensor):
        images = [images[i] for i in range(len(images))]

    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (n_cols * 2.5, n_rows * 2.5)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes[np.newaxis, :] if n_cols > 1 else np.array([[axes]])
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx in range(n_rows * n_cols):
        ax = axes[idx // n_cols, idx % n_cols]
        if idx < n_images:
            img = tensor_to_numpy(images[idx])
            ax.imshow(img)
        ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardee : {save_path}")

    plt.show()


def show_comparison(real_images, generated_images, n_samples=4,
                    labels=('Original', 'Genere'), title=None, save_path=None):
    """
    Affiche une comparaison cote-a-cote entre images reelles et generees.

    Disposition :
        Original_1  | Genere_1
        Original_2  | Genere_2
        ...

    Args:
        real_images: tensor (N, C, H, W)
        generated_images: tensor (N, C, H, W)
        n_samples: nombre de paires a afficher
        labels: noms des colonnes
        title: titre de la figure
        save_path: chemin de sauvegarde
    """
    n_samples = min(n_samples, len(real_images), len(generated_images))
    fig, axes = plt.subplots(n_samples, 2, figsize=(6, n_samples * 2.5))

    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_samples):
        # Image reelle
        real_np = tensor_to_numpy(real_images[i])
        axes[i, 0].imshow(real_np)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title(labels[0], fontsize=12)

        # Image generee
        gen_np = tensor_to_numpy(generated_images[i])
        axes[i, 1].imshow(gen_np)
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title(labels[1], fontsize=12)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardee : {save_path}")

    plt.show()


def show_cyclegan_results(real_a, fake_b, cycle_a, real_b, fake_a, cycle_b,
                          n_samples=3, save_path=None):
    """
    Affiche les resultats du CycleGAN avec le cycle complet.

    Disposition :
        Real A  |  Fake B  |  Cycle A  ||  Real B  |  Fake A  |  Cycle B

    Permet de verifier visuellement :
      - La qualite de la transformation (Fake B coherent ?)
      - La coherence cyclique (Cycle A ≈ Real A ?)
    """
    n_samples = min(n_samples, len(real_a))
    fig, axes = plt.subplots(n_samples, 6, figsize=(18, n_samples * 2.5))

    if n_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Real A\n(normal)', 'Fake B\n(secheresse)', 'Cycle A\n(reconstruit)',
                  'Real B\n(secheresse)', 'Fake A\n(normal)', 'Cycle B\n(reconstruit)']

    for i in range(n_samples):
        tensors = [real_a[i], fake_b[i], cycle_a[i], real_b[i], fake_a[i], cycle_b[i]]
        for j, tensor in enumerate(tensors):
            img = tensor_to_numpy(tensor)
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=10)

    plt.suptitle('Resultats CycleGAN : cycle complet', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def show_ndvi_comparison(images_normal, images_drought, n_samples=3, save_path=None):
    """
    Compare les cartes NDVI entre images normales et en secheresse.

    Le NDVI (proxy RGB) devrait etre plus eleve pour les images normales
    (vegetation saine = beaucoup de vert) et plus bas pour la secheresse.
    """
    n_samples = min(n_samples, len(images_normal), len(images_drought))
    fig, axes = plt.subplots(n_samples, 4, figsize=(14, n_samples * 3))

    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_samples):
        # Image normale
        axes[i, 0].imshow(tensor_to_numpy(images_normal[i]))
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Normal', fontsize=11)

        # NDVI normal
        ndvi_normal = compute_ndvi_proxy(images_normal[i]).cpu().numpy()
        im1 = axes[i, 1].imshow(ndvi_normal, cmap='RdYlGn', vmin=-0.5, vmax=0.5)
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title('NDVI Normal', fontsize=11)

        # Image secheresse
        axes[i, 2].imshow(tensor_to_numpy(images_drought[i]))
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 2].set_title('Secheresse', fontsize=11)

        # NDVI secheresse
        ndvi_drought = compute_ndvi_proxy(images_drought[i]).cpu().numpy()
        im2 = axes[i, 3].imshow(ndvi_drought, cmap='RdYlGn', vmin=-0.5, vmax=0.5)
        axes[i, 3].axis('off')
        if i == 0:
            axes[i, 3].set_title('NDVI Secheresse', fontsize=11)

    plt.suptitle('Comparaison NDVI : Normal vs Secheresse', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_training_losses(history, title='Courbes de perte', save_path=None):
    """
    Affiche les courbes de perte pendant l'entrainement.

    Args:
        history: dict avec les cles = nom de la perte, valeurs = listes
        title: titre du graphique
        save_path: chemin de sauvegarde
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for name, values in history.items():
        ax.plot(values, label=name, linewidth=1.5)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Perte', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_metrics_summary(metrics_dict, save_path=None):
    """
    Affiche un resume des metriques d'evaluation sous forme de barres.

    Args:
        metrics_dict: dict {'SSIM': val, 'PSNR': val, 'FID': val}
        save_path: chemin de sauvegarde
    """
    fig, axes = plt.subplots(1, len(metrics_dict), figsize=(4 * len(metrics_dict), 4))

    if len(metrics_dict) == 1:
        axes = [axes]

    for ax, (name, value) in zip(axes, metrics_dict.items()):
        color = '#2196F3' if name != 'FID' else '#FF5722'
        ax.bar([name], [value], color=color, width=0.5)
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_ylabel('Score')
        ax.text(0, value, f'{value:.2f}', ha='center', va='bottom', fontsize=12)

    plt.suptitle('Metriques d\'evaluation', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
