"""
dataset.py
----------
Chargement du dataset EuroSAT, filtrage des classes agricoles,
et creation des DataLoaders pour l'entrainement.

EuroSAT contient 27 000 images Sentinel-2 en 64x64 pixels, reparties
en 10 classes de couverture terrestre. On ne garde que les classes
agricoles definies dans config.py.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

from src.config import (
    RAW_DIR, AGRICULTURAL_CLASSES, IMAGE_SIZE,
    NORMALIZE_MEAN, NORMALIZE_STD, SPLIT_RATIOS, RANDOM_SEED,
)


class EuroSATAgricultural(Dataset):
    """
    Dataset EuroSAT filtre pour ne garder que les classes agricoles.

    Telecharge automatiquement EuroSAT si necessaire via torchvision,
    puis filtre les images pour ne conserver que les classes definies
    dans AGRICULTURAL_CLASSES.
    """

    def __init__(self, root=RAW_DIR, transform=None, download=True):
        """
        Args:
            root: repertoire ou stocker/charger EuroSAT
            transform: transformations a appliquer
            download: telecharger si pas deja present
        """
        self.transform = transform

        # On charge le dataset complet via torchvision.
        # EuroSAT est organise en sous-dossiers par classe.
        full_dataset = datasets.EuroSAT(
            root=root,
            download=download,
        )

        # Mapping nom_classe -> index dans EuroSAT
        # EuroSAT utilise l'ordre alphabetique des dossiers.
        class_to_idx = full_dataset.class_to_idx
        self.selected_classes = AGRICULTURAL_CLASSES

        # Indices des classes qu'on veut garder
        target_indices = set()
        self.class_names = []
        for cls_name in self.selected_classes:
            if cls_name in class_to_idx:
                target_indices.add(class_to_idx[cls_name])
                self.class_names.append(cls_name)
            else:
                print(f"Attention : classe '{cls_name}' non trouvee dans EuroSAT")

        # Filtrer les echantillons pour ne garder que nos classes
        self.samples = []
        self.labels = []
        # Re-indexation des labels (0, 1, 2, ...) pour nos classes
        old_to_new = {}
        for new_idx, cls_name in enumerate(self.class_names):
            old_to_new[class_to_idx[cls_name]] = new_idx

        for path, label in full_dataset.samples:
            if label in target_indices:
                self.samples.append(path)
                self.labels.append(old_to_new[label])

        print(f"EuroSAT agricole : {len(self.samples)} images, "
              f"{len(self.class_names)} classes {self.class_names}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class UnpairedDomainDataset(Dataset):
    """
    Dataset pour CycleGAN : paires non appariees de deux domaines.

    Domaine A : images agricoles normales
    Domaine B : images agricoles transformees en "secheresse"

    Comme CycleGAN est non supervise, les images de A et B n'ont
    pas besoin d'etre alignees. On pioche aleatoirement dans chaque domaine.
    """

    def __init__(self, domain_a_dir, domain_b_dir, transform=None):
        """
        Args:
            domain_a_dir: dossier contenant les images du domaine A (normal)
            domain_b_dir: dossier contenant les images du domaine B (secheresse)
            transform: transformations communes
        """
        self.transform = transform

        # Lister les fichiers images dans chaque domaine
        valid_ext = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        self.files_a = sorted([
            os.path.join(domain_a_dir, f) for f in os.listdir(domain_a_dir)
            if os.path.splitext(f)[1].lower() in valid_ext
        ])
        self.files_b = sorted([
            os.path.join(domain_b_dir, f) for f in os.listdir(domain_b_dir)
            if os.path.splitext(f)[1].lower() in valid_ext
        ])

        print(f"Domaine A (normal) : {len(self.files_a)} images")
        print(f"Domaine B (secheresse) : {len(self.files_b)} images")

    def __len__(self):
        # On prend le max pour ne pas perdre d'images du plus grand domaine
        return max(len(self.files_a), len(self.files_b))

    def __getitem__(self, idx):
        # Acces cyclique si les domaines n'ont pas la meme taille
        img_a = Image.open(self.files_a[idx % len(self.files_a)]).convert('RGB')
        img_b = Image.open(self.files_b[idx % len(self.files_b)]).convert('RGB')

        if self.transform is not None:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)

        return img_a, img_b


def get_agricultural_dataset(transform=None, download=True):
    """Charge le dataset EuroSAT filtre sur les classes agricoles."""
    return EuroSATAgricultural(
        root=RAW_DIR, transform=transform, download=download
    )


def split_dataset(dataset, seed=RANDOM_SEED):
    """
    Separe un dataset en train / val / test selon les ratios de config.
    Retourne trois Subsets.
    """
    n_total = len(dataset)
    n_train = int(n_total * SPLIT_RATIOS['train'])
    n_val = int(n_total * SPLIT_RATIOS['val'])
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    print(f"Split : train={n_train}, val={n_val}, test={n_test}")
    return train_set, val_set, test_set


def get_dataloader(dataset, batch_size, shuffle=True, num_workers=2):
    """Cree un DataLoader avec les parametres par defaut."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
