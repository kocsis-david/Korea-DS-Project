# Quick Use — FungiCLEF 2025 pipeline

Reproduce `submission_ensemble.csv` (Kaggle private 0.79 / public 0.82) from scratch.
For the *why* behind the design, see `doc.md`; for the report, see `PRESENTATION.md`.

---

## 0. Prerequisites
- **Python 3.12**, ~16 GB RAM, ~12 GB free disk (dataset + caches).
- **GPU strongly recommended** (feature extraction is the only heavy step). Tested on
  an RTX 5090 / CUDA 12.8; CPU works but is slow.
- A **Kaggle account** + API token to download the data.

---

## 1. Install
```bash
# GPU PyTorch first (match your CUDA — example is CUDA 12.8):
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
# then the rest:
pip install -r requirements.txt
```
The two foundation models download automatically on first run from the Hugging Face Hub
(BioCLIP-2 ≈ 1 GB, DINOv3-B ≈ 0.35 GB) — needs internet once, then cached by `huggingface_hub`.

---

## 2. Get the data
Download the competition data into a local `data/` folder:
```bash
# uses the Kaggle CLI 2.x token (env var or ~/.kaggle/access_token)
export KAGGLE_API_TOKEN=KGAT_xxx          # your token (also stored in .env)
kaggle competitions download -c fungi-clef-2025 -p data/
cd data && unzip -q '*.zip' && cd ..
```

Expected layout (the scripts read these paths):
```
data/
├── images/FungiTastic-FewShot/{train,val,test}/500p/*.JPG
├── metadata/FungiTastic-FewShot/FungiTastic-FewShot-{Train,Val,Test}.csv
├── captions/{train,val,test}/*.JPG.json          # only for fungi_captions.py
└── FungiCLEF25-SAMPLE_SUBMISSION.csv
```

---

## 3. Run (two commands)
```bash
python fungi_features.py     # 1) extract + cache BioCLIP-2 & DINOv3 embeddings (+ hflip TTA)
python fungi_train.py        # 2) train heads, honest CV, write submission_ensemble.csv
```
- Step 1 is the only GPU-heavy part (~a few minutes on a modern GPU for ~12k images ×2
  backbones ×2 for TTA). Results are cached in `features_ensemble/*.npy`, so **re-runs are
  instant** — delete that folder to force re-extraction.
- Step 2 runs in seconds (no backbone training).

**Output:** `submission_ensemble.csv` — 999 rows, columns `observationId,predictions`
(top-10 `category_id`s, space-separated). This is the file you upload to Kaggle.

`fungi_train.py` also prints the honest, leakage-free validation estimate:
```
+ transductive  : top1≈0.530  top5≈0.754  top10≈0.832
```

---

## 4. Or: the notebook (same result + figures)
```bash
jupyter notebook fungi_ensemble.ipynb     # run all cells
# headless:
python -m nbconvert --to notebook --execute --inplace fungi_ensemble.ipynb
```
Produces the four `viz_ens_*.png` figures and the submission.

---

## 5. Files at a glance
| File | Role |
|---|---|
| `fungi_features.py` | cached BioCLIP-2 + DINOv3 embeddings (500px) + hflip TTA |
| `fungi_train.py` | metadata encoder, centroid / kNN / **soft-transductive** heads, ensemble grid-search, observation-disjoint CV, **writes submission** |
| `fungi_captions.py` | BioCLIP-2 text embeddings of captions (tested, unused — kept for transparency) |
| `fungi_ensemble.ipynb` | narrated end-to-end driver with visualisations |
| `submission_ensemble.csv` | the Kaggle upload |

---

## 6. Troubleshooting
- **`OSError: image file is truncated`** — already handled (`LOAD_TRUNCATED_IMAGES`); if you
  see it elsewhere, a few dataset images are byte-truncated and safely padded.
- **CUDA OOM** — lower `batch_size` in `fungi_features.py` (`extract` / `extract_flip`).
- **Slow / CPU-only** — confirm `python -c "import torch; print(torch.cuda.is_available())"`
  prints `True`; if not, reinstall the CUDA torch wheel (step 1).
- **HF download blocked** — set `HF_HOME` to a writable cache, or pre-download
  `imageomics/bioclip-2` and `timm/vit_base_patch16_dinov3.lvd1689m`.
- **Different paths** — edit `DATA_ROOT` / `IMG_RES` at the top of `fungi_features.py`.
