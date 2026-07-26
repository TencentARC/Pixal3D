# Artefacts RunPod Pixal3D

Sauvegarde ciblée de l’environnement CUDA utilisé pour Pixal3D sur RunPod.

Les archives binaires sont volontairement conservées uniquement en local :
elles sont volumineuses et ne sont pas nécessaires au portage macOS/MPS.

## Contenu

- `pixal3d-cuda-binaries.tar.gz` — 669 Mo (local uniquement) : NATTEN, FlashAttention, `o_voxel`, `cumesh` et `flex_gemm`.
- `pixal3d-cuda-extras.tar.gz` — 3,9 Mo (local uniquement) : `nvdiffrast` et `nvdiffrec_render`.
- `pixal3d-python-lock.txt` — versions exactes observées dans `/workspace/venv`.

Les modèles Hugging Face, les caches et le dépôt Pixal3D ne sont pas inclus.

## Compatibilité

Les archives ciblent l’environnement Linux suivant : Python 3.11, PyTorch 2.6.0 + CUDA 12.4. Elles ne sont pas utilisables directement dans l’environnement macOS/MPS.

## Restauration

Sur un nouveau pod compatible, installer d’abord Python et les dépendances générales, puis extraire les extensions dans l’environnement virtuel :

```bash
SITE=/workspace/venv/lib/python3.11/site-packages
tar -xzf pixal3d-cuda-binaries.tar.gz -C "$SITE"
tar -xzf pixal3d-cuda-extras.tar.gz -C "$SITE"
```

Le fichier de lock contient quelques références locales au pod (`file:///tmp/...` et `file:///workspace/...`) ; il sert de référence et de contrôle, mais ne doit pas être utilisé tel quel comme unique commande `pip install -r`.
