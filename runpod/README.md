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

## Restore checks found during CUDA validation (2026-09-12)

The original extras archive omits the top-level
`_nvdiffrast_c.cpython-311-x86_64-linux-gnu.so` extension. Extracting only the
two original archives is therefore insufficient. A complete replacement,
`pixal3d-nvdiffrast-complete-torch260-cu128-sm86.tar.gz`, is retained locally
and must be extracted **after** the original extras archive.

This replacement was built from NVIDIA nvdiffrast v0.4.0
(`253ac4fcea7de5f396371124af597e6cc957bfae`) with Python 3.11, PyTorch
2.6.0+cu124, the CUDA 12.8 compiler and `TORCH_CUDA_ARCH_LIST=8.6`.
Its tested target is an A40 with driver 580.159.04; compatibility with other
architectures or older drivers is not established.

SHA-256: `ce0dd6009d504570bc07b393b78e66b53dca0141cf5e35deb3d8a5bbc709cb2e`.

MoGe's pinned package metadata requests a different `utils3d` source than
the saved environment. When restoring with `uv`, explicitly override that
dependency with the `utils3d` wheel recorded in the lock, and pin PyTorch
and torchvision to their saved CUDA versions.

Do not copy an entire Hugging Face home directory to a pod: it can contain
personal access tokens. Transfer only specifically authorized model files
when a required gated model cannot be downloaded anonymously.
