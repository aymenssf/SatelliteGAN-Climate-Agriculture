"""
config.py
---------
Hyperparametres centralises pour l'ensemble du projet.
Chaque parametre est documente et justifie.

On utilise un dictionnaire simple plutot qu'un systeme de config
elabore (argparse, hydra...) -- c'est suffisant pour un projet
de cette taille et ca reste lisible.
"""

import torch
import os

# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------------------------------------------
# Chemins
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
CYCLEGAN_CKPT_DIR = os.path.join(OUTPUT_DIR, 'cyclegan_checkpoints')
DIFFUSION_CKPT_DIR = os.path.join(OUTPUT_DIR, 'diffusion_checkpoints')
GENERATED_DIR = os.path.join(OUTPUT_DIR, 'generated_images')

# ------------------------------------------------------------------
# Dataset EuroSAT
# ------------------------------------------------------------------
# On selectionne 4 classes liees a l'agriculture :
#   - AnnualCrop : cultures annuelles (ble, mais...)
#   - PermanentCrop : cultures permanentes (vignes, vergers)
#   - Pasture : prairies / paturages
#   - HerbaceousVegetation : vegetation herbacee naturelle
#
# Raison : ces classes representent les zones agricoles et vegetales
# les plus susceptibles d'etre impactees par la secheresse.
# Les autres classes (foret, riviere, zone urbaine...) ne sont pas
# pertinentes pour notre scenario de stress hydrique agricole.
AGRICULTURAL_CLASSES = [
    'AnnualCrop',
    'PermanentCrop',
    'Pasture',
    'HerbaceousVegetation',
]

# Taille des images en entree.
# 64x64 est un compromis entre qualite visuelle et contraintes GPU Colab.
# EuroSAT est a 64x64 nativement, donc pas de perte d'information.
IMAGE_SIZE = 64

# Normalisation : valeurs moyennes et ecarts-types approximatifs
# pour les 3 bandes RGB de Sentinel-2 (estimees sur EuroSAT).
# On normalise dans [-1, 1] pour le GAN (convention standard).
NORMALIZE_MEAN = [0.5, 0.5, 0.5]
NORMALIZE_STD = [0.5, 0.5, 0.5]

# ------------------------------------------------------------------
# CycleGAN
# ------------------------------------------------------------------
CYCLEGAN = {
    # Nombre de blocs residuels dans le generateur.
    # 6 blocs : bon compromis pour des images 64x64.
    # (9 blocs serait plus adapte pour 256x256, cf. papier original)
    'n_residual_blocks': 6,

    # Taille du batch. 4 est un bon compromis pour la RAM Colab T4 (16 GB).
    # Les GANs profitent de petits batchs pour la diversite des gradients.
    'batch_size': 4,

    # Learning rate. 2e-4 est la valeur standard du papier CycleGAN.
    'lr': 2e-4,

    # Betas pour Adam. (0.5, 0.999) est la convention pour les GANs
    # (le papier DCGAN recommande beta1=0.5 au lieu de 0.9).
    'betas': (0.5, 0.999),

    # Poids des differentes losses :
    # lambda_cycle : importance de la coherence cyclique.
    # 10.0 est la valeur du papier original. Ca force le generateur
    # a ne pas inventer n'importe quoi et a conserver la structure.
    'lambda_cycle': 10.0,

    # lambda_identity : regularisation optionnelle.
    # Si on donne une image deja dans le domaine cible au generateur,
    # il devrait la laisser inchangee. Aide a preserver les couleurs.
    # 5.0 = 0.5 * lambda_cycle, comme dans le papier.
    'lambda_identity': 5.0,

    # Nombre d'epochs. 100 est raisonnable sur Colab (~3-5h sur T4).
    'n_epochs': 100,

    # Epoch a partir duquel on commence le decay lineaire du lr.
    'decay_epoch': 50,

    # Sauvegarder un checkpoint tous les N epochs.
    'save_every': 10,

    # Taille du buffer de replay pour le discriminateur.
    # Technique standard pour stabiliser l'entrainement GAN.
    'replay_buffer_size': 50,
}

# ------------------------------------------------------------------
# Diffusion (DDPM)
# ------------------------------------------------------------------
DIFFUSION = {
    # Nombre de pas de diffusion (T dans le papier).
    # 1000 est la valeur standard du papier DDPM.
    'n_timesteps': 1000,

    # Schedule du bruit. 'linear' est le choix original de Ho et al.
    'schedule': 'linear',

    # Bornes du schedule lineaire (beta_start, beta_end).
    # Valeurs standard du papier DDPM.
    'beta_start': 1e-4,
    'beta_end': 0.02,

    # Architecture U-Net (compatible checkpoints existants)
    'base_channels': 64,
    'channel_mults': (1, 2, 4),
    'n_res_blocks': 2,
    'attention_levels': [2],

    # Entrainement
    'batch_size': 16,
    'lr': 1e-4,               # Reduit (etait 2e-4) pour plus de stabilite
    'n_epochs': 150,

    # Sampling DDIM : 100 steps deterministes ~ 1000 steps DDPM en qualite.
    'sampling_steps': 100,

    # Sauvegarde
    'save_every': 25,
}

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
EVAL = {
    # Nombre d'images a generer pour l'evaluation.
    # 1000 minimum pour un FID fiable (N > D avec InceptionV3 2048-D).
    'n_generated': 1000,

    # Nombre d'images pour le calcul FID (si utilise).
    'fid_n_samples': 1000,
}

# ------------------------------------------------------------------
# Split train/val/test
# ------------------------------------------------------------------
SPLIT_RATIOS = {
    'train': 0.8,
    'val': 0.1,
    'test': 0.1,
}

# ------------------------------------------------------------------
# Seed pour la reproductibilite
# ------------------------------------------------------------------
RANDOM_SEED = 42
