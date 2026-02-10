# Prediction des Impacts du Changement Climatique sur l'Agriculture par Imagerie Satellite

## Contexte

Le changement climatique modifie progressivement les conditions d'exploitation agricole
en Europe. Les episodes de secheresse, de plus en plus frequents, affectent directement
le rendement des cultures et la sante de la vegetation. L'imagerie satellite (Sentinel-2)
offre un moyen d'observation a grande echelle, mais les donnees de scenarios futurs
n'existent pas encore -- il faut les generer.

Ce projet explore deux approches complementaires de generation d'images satellites
pour simuler l'impact visuel de stress hydrique sur les parcelles agricoles :

1. **CycleGAN** pour l'adaptation de domaine (images normales vers images secheresse)
2. **Modele de diffusion (DDPM)** pour la generation de scenarios climatiques synthetiques

## Objectifs

- Charger et analyser le dataset EuroSAT (Sentinel-2, 10 classes de couverture terrestre)
- Selectionner les classes agricoles pertinentes et justifier ce choix
- Construire un domaine "secheresse" synthetique par transformation spectrale
- Entrainer un CycleGAN pour apprendre la transformation normal-secheresse
- Entrainer un modele de diffusion (DDPM) sur les images transformees
- Evaluer les resultats qualitativement (visualisations) et quantitativement (SSIM, PSNR, FID)

## Methodologie

Le pipeline se decompose en quatre etapes :

```
EuroSAT (Sentinel-2)
    |
    v
[Selection classes agricoles] --> AnnualCrop, PermanentCrop, Pasture, HerbaceousVegetation
    |
    v
[Simulation domaine secheresse]  --> Transformations spectrales (NDVI, bandes RGB)
    |
    v
[CycleGAN]  Normal <--> Secheresse
    |
    v
[DDPM Diffusion]  Generation de scenarios synthetiques
    |
    v
[Evaluation]  SSIM / PSNR / FID + analyse visuelle
```

Le CycleGAN apprend un mapping bidirectionnel entre images agricoles normales et
leur equivalent en conditions de secheresse. Le modele de diffusion est ensuite
entraine sur les images du domaine secheresse pour generer de nouveaux echantillons
realistes simulant des conditions de stress hydrique.

## Structure du projet

```
climate-agriculture-gan/
|
+-- README.md
+-- requirements.txt
+-- .gitignore
+-- setup_colab.py
|
+-- notebooks/
|   +-- 01_data_exploration.ipynb
|   +-- 02_cyclegan_training.ipynb
|   +-- 03_diffusion_training.ipynb
|   +-- 04_evaluation_results.ipynb
|
+-- src/
|   +-- __init__.py
|   +-- config.py
|   +-- dataset.py
|   +-- preprocessing.py
|   |
|   +-- cyclegan/
|   |   +-- __init__.py
|   |   +-- models.py
|   |   +-- losses.py
|   |   +-- train.py
|   |   +-- utils.py
|   |
|   +-- diffusion/
|   |   +-- __init__.py
|   |   +-- unet.py
|   |   +-- scheduler.py
|   |   +-- diffusion_model.py
|   |   +-- train.py
|   |
|   +-- evaluation/
|       +-- __init__.py
|       +-- metrics.py
|       +-- visualization.py
|
+-- outputs/              (genere localement, ignore par git)
+-- data/                 (telecharge localement, ignore par git)
```

## Installation et execution

### Option 1 : Google Colab (recommande)

1. Cloner le depot :
   ```
   !git clone https://github.com/aymenssf/SatelliteGAN-Climate-Agriculture.git
   %cd SatelliteGAN-Climate-Agriculture
   !pip install -q -r requirements.txt
   ```

2. Ouvrir les notebooks dans l'ordre :
   - `01_data_exploration.ipynb` -- exploration des donnees, selection des classes
   - `02_cyclegan_training.ipynb` -- entrainement du CycleGAN (~3-5h sur T4)
   - `03_diffusion_training.ipynb` -- entrainement du modele de diffusion (~4-6h sur T4)
   - `04_evaluation_results.ipynb` -- evaluation et visualisations finales

3. Chaque notebook contient une cellule de setup en debut. Monter Google Drive
   pour sauvegarder les checkpoints en cas de deconnexion.

### Option 2 : Execution locale

```bash
git clone https://github.com/aymenssf/SatelliteGAN-Climate-Agriculture.git
cd SatelliteGAN-Climate-Agriculture
pip install -r requirements.txt
```

Un GPU est fortement recommande. L'execution CPU est possible mais tres lente.

## Resultats attendus

- Le CycleGAN produit des transformations visuellement coherentes : les parcelles
  agricoles normales sont converties en versions "seches" avec reduction de la
  vegetation verte, jaunissement, et modification des indices spectraux.

- Le modele de diffusion genere des images satellites synthetiques de secheresse
  avec une diversite suffisante pour simuler differents niveaux de stress hydrique.

- Les metriques SSIM et PSNR permettent de verifier que les transformations
  conservent la structure spatiale des images (routes, limites de parcelles).

## Limitations et perspectives

**Limitations :**

- Le domaine "secheresse" est simule par transformation spectrale, pas a partir
  de vraies images de secheresse. Cela introduit un biais dans l'apprentissage.
- Le modele de diffusion travaille en pixel space (pas de VAE latent), ce qui
  limite la resolution et la qualite par rapport a un vrai Latent Diffusion Model.
- L'evaluation FID est limitee par la taille du dataset genere (quelques centaines
  d'images vs les milliers recommandes pour un FID fiable).
- Le temps d'entrainement sur GPU Colab gratuit contraint la taille des modeles
  et le nombre d'epochs.

**Perspectives :**

- Utiliser de vraies images de secheresse (Copernicus Emergency Management Service)
  pour un apprentissage plus realiste.
- Passer a un veritable Latent Diffusion Model avec VAE pre-entraine.
- Integrer un conditionnement plus fin (niveau de severite de secheresse, saison).
- Valider les resultats avec des agronomes pour evaluer le realisme agronomique.

## References

- Zhu, J.-Y., et al. (2017). "Unpaired Image-to-Image Translation using
  Cycle-Consistent Adversarial Networks." ICCV 2017.
- Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic
  Models." NeurIPS 2020.
- Helber, P., et al. (2019). "EuroSAT: A Novel Dataset and Deep Learning
  Benchmark for Land Use and Land Cover Classification." IEEE JSTARS.
- Isola, P., et al. (2017). "Image-to-Image Translation with Conditional
  Adversarial Networks." CVPR 2017.
- Nichol, A. & Dhariwal, P. (2021). "Improved Denoising Diffusion Probabilistic
  Models." ICML 2021.
- Mao, X., et al. (2017). "Least Squares Generative Adversarial Networks." ICCV 2017.
