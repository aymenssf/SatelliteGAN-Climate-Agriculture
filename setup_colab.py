"""
setup_colab.py
--------------
Script de configuration pour Google Colab.
A executer en premiere cellule de chaque notebook pour preparer
l'environnement (Drive, packages, GPU).

Usage dans un notebook Colab :
    %run setup_colab.py
"""

import os
import subprocess
import sys


def mount_drive():
    """Monte Google Drive pour la sauvegarde des checkpoints."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive monte avec succes.")
        return True
    except ImportError:
        print("Pas sur Colab, Drive non monte.")
        return False


def install_requirements():
    """Installe les dependances depuis requirements.txt."""
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-q', '-r', req_path
        ])
        print("Dependances installees.")
    else:
        print(f"requirements.txt non trouve a {req_path}")


def check_gpu():
    """Verifie la disponibilite du GPU et affiche les infos."""
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU disponible : {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("Aucun GPU detecte. L'entrainement sera tres lent.")
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_directories():
    """Cree les dossiers de sortie s'ils n'existent pas."""
    dirs = [
        'outputs/cyclegan_checkpoints',
        'outputs/diffusion_checkpoints',
        'outputs/generated_images',
        'data/raw',
        'data/processed',
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Dossiers de sortie crees.")


def setup():
    """Configuration complete de l'environnement Colab."""
    print("=" * 50)
    print("Configuration de l'environnement")
    print("=" * 50)

    is_colab = mount_drive()
    install_requirements()
    device = check_gpu()
    create_directories()

    print("=" * 50)
    print("Setup termine.")
    print("=" * 50)
    return device


if __name__ == '__main__':
    setup()
