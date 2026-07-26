# Pixal3D sur macOS Apple Silicon

Ce dépôt contient le code officiel Pixal3D et un port d’exécution MPS pour le
Mac M3 Max 36 Go. Le port reprend les backends Metal validés dans
`../trellis2-macos` : `mtldiffrast`, `mtlmesh`, `mtlgemm`, `mtlbvh` et le fork
Apple de `o_voxel`.

## Installation

```bash
xcodebuild -downloadComponent MetalToolchain
bash setup_macos.sh
source .venv/bin/activate
```

Les poids sont téléchargés à la demande par Hugging Face. Pour les mettre en
cache avant le premier calcul :

```bash
python scripts/download_models.py
```

## Génération iso-qualité CUDA

Le profil `cuda-parity` conserve la cascade neurale 1536, le volume PBR
1536, une cible d’environ un million de faces et les textures PBR 4096 px.
Le dual-contouring utilise par défaut une grille 512³ : c’est le profil
validé avec marge sur 36 Go, et son résultat passe les contrôles structuraux
face au GLB CUDA de référence.

```bash
python inference.py \
  --image assets/images/0_img.png \
  --output output/0_cuda_parity.glb \
  --low_vram \
  --resolution 1536 \
  --export-profile cuda-parity
```

Avant l’export, le programme sauvegarde automatiquement
`output/0_cuda_parity.decoded.pt`. Ce checkpoint contient le maillage décodé
et son volume PBR sparse, sans les poids des modèles. Si le remeshing ou la
texture 4096 échoue, l’export peut être repris sans relancer la génération :

```bash
python inference.py \
  --image assets/images/0_img.png \
  --decoded-checkpoint output/0_cuda_parity.decoded.pt \
  --output output/0_cuda_parity.glb \
  --export-profile cuda-parity
```

`--remesh-resolution` permet d’expérimenter avec une grille plus dense, mais
la mémoire du simplificateur croît rapidement au-delà de 512. Le profil
historique léger reste accessible avec `--export-profile portable`.

La pression sur les 36 Go de mémoire unifiée est limitée de quatre façons :

- DINOv3 et NAF sont partagés entre les quatre conditionneurs ;
- chaque flow/decoder est supprimé après sa dernière étape ;
- le maillage et le volume décodés passent sur CPU avant l’export ;
- le BVH source de 18 millions de triangles est réduit exactement sur des
  hiérarchies successives de 250 000 faces ;
- le remesh, le nettoyage/simplification et le bake sont des étapes séparées,
  ce qui libère leurs allocations Metal entre elles ;
- l’échantillonnage du volume est découpé en lots ;
- seuls les sommets d’échantillonnage puis les rares texels invalides sont
  reprojetés sur la surface source.

La grille dual-contouring n’est volontairement pas tuilée : des blocs
indépendants créeraient des raccords. C’est le BVH de distance qui est
découpé, puis réduit par minimum global, donc sans fissure aux frontières.

### Validation de référence

Le cas `output/inputs/0_img_2048.png`, seed 42, a été comparé au GLB produit
par la branche `main` sur une RTX A5000 RunPod :

- CUDA : 937 343 faces, 5 arêtes de bord, 99,47 % dans la composante
  principale, aire 5,197 ;
- MPS final : 989 941 faces, 7 arêtes de bord, 99,46 % dans la composante
  principale, aire 4,944 ;
- les deux fichiers ont une texture 4096², un matériau opaque/simple face et
  un alpha p01 de 254.

Le contrôle automatisé se relance avec :

```bash
python -m scripts.compare_glb_quality \
  --reference output/pixal3d_main_cuda_a5000_2048_lowvram.glb \
  --candidate output/pixal3d_mps_1536_cuda_parity_final.glb
```

Sur ce Mac, la génération neurale 1536 mesurée prend 1 979,85 s et l’export
final intégré 128,65 s, soit environ 35 min 09 s au total. Le même cas avait
pris environ 13 min 25 s sur l’A5000.

## Interface

Pour lancer l’interface locale :

```bash
python app.py --low_vram
```

Le CLI utilise les convolutions sparse `flex_gemm`, SDPA sur MPS pour les
longues séquences et le vrai modèle NAF appris. Le noyau d’attention Metal
fusionné reste disponible, mais il n’est pas retenu par défaut : il régresse
fortement vers 40 000 tokens malgré ses bons résultats sur les petites
séquences. Seule l’opération NATTEN CUDA de NAF est remplacée par une
implémentation MPS équivalente, découpée par lignes. Ces chemins ont des tests
numériques FP16/BF16 face à leurs références PyTorch.

Les ajouts spécifiques sont listés dans `requirements-macos.txt`. Le setup
retient la version récente de `utils3d` requise par MoGe ; la wheel 0.0.2
indiquée dans la fiche Pixal3D est trop ancienne pour cette API.
`requirements-hfdemo.txt` est réservé au Space Hugging Face et ne doit pas
être utilisé ici.
