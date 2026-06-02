# FungiCLEF 2025 — Solution Documentation

Few-shot fungal species identification on the **FungiTastic–FewShot** dataset.

- **Classes:** 2427 species
- **Shots:** ~3.2 training images per class (extreme few-shot)
- **Test:** 1911 images grouped into **999 observations** (one prediction per observation)
- **Submission:** for each observation, a ranked list of the **top-10** `category_id`s

**Final result (honest, leakage-free estimate):** top-1 ≈ **53%**, top-5 ≈ **75%**, top-10 ≈ **83%**
— a ~6× jump over the original fine-tuned baselines (~5–8% top-1).

---

## 1. Core idea — why this design

With only ~3 images per class, **fine-tuning a network destroys more than it learns**:
gradient updates wreck the pretrained features faster than 2427 classes can be learned.
We confirmed this — a CAFormer fine-tune collapsed to ~0% (random) after one epoch.

The robust recipe for extreme few-shot is the opposite:

> **Frozen foundation-model embeddings + a non-parametric (prototype) head.**

Nothing in the backbone moves, so the powerful pretrained features are preserved and
there is no deep head to overfit. Training is effectively instantaneous (no backprop
through the backbone). On top of this base we layer three accuracy levers that each
earned their place by **measurement**, not assumption.

---

## 2. The pipeline

```
                 ┌────────────────────────────────────────────┐
  500px image →  │ BioCLIP-2 (frozen)  ─┐                       │
                 │   + hflip TTA        ├─ L2-norm embedding ───┼─► cosine-centroid
                 │ DINOv3 (frozen)     ─┘  (768-d each)         │   prototype head
                 └────────────────────────────────────────────┘        │
  metadata    →  cyclic month + one-hot habitat/substrate/… ───────────►│ (re-rankers,
  captions    →  BioCLIP-2 text tower  (tested — dropped, see §5)       │  down-weighted)
                                                                        ▼
                          soft transductive prototype adaptation (test-time)
                                                                        ▼
                          per-observation mean → top-10 species → submission
```

### Backbones (both frozen)
| Backbone | Source | Input | Why |
|---|---|---|---|
| **BioCLIP-2** | `open_clip` · `hf-hub:imageomics/bioclip-2` | 224px, CLIP-norm | Trained on the Tree of Life — **in-domain** for species ID. The workhorse. |
| **DINOv3-B/16** | `timm` · `vit_base_patch16_dinov3.lvd1689m` | 256px, ImageNet-norm | Best *general* self-supervised ViT — included as a complementary signal. |

Embeddings are extracted from the **500px** images (more fine-grained texture than the
300px baseline) and L2-normalised. Every pass is cached to `features_ensemble/` so the
heavy GPU work runs only once.

### Head — cosine-centroid (prototype)
For each class, average its (L2-normalised) training embeddings into a centroid;
classify by cosine similarity to the centroids. No parameters, cannot overfit. A soft
k-NN variant and a small metadata classifier are also built as optional re-rankers.

### Three accuracy levers (each measured on a leakage-free val split)
| Lever | What it does | Gain (top-10) |
|---|---|---|
| **1. hflip TTA** | Average the original + horizontally-flipped embedding | small, free |
| **2. train + val reference** | Fold the **labelled val set** into the prototype pool for the test predictions | **+5–6 pts** — the biggest single win |
| **3. soft transductive prototypes** | A few EM steps that nudge each centroid toward the unlabelled **test** distribution using probability-weighted votes | **+1.3 pts** |

The decisive lever is **#2**: val has ~2285 labelled images we are free to use as
reference data (only the test labels are hidden), roughly doubling the shots-per-class
available at inference.

---

## 3. Honest evaluation methodology

Everything is evaluated **at the observation level** (mean of an observation's
per-image probabilities), matching the competition's one-prediction-per-observation rule.

To estimate test performance *before* submitting, we run an **observation-disjoint
2-fold CV on val**: each eval fold is predicted using `train + the other val fold` as
reference. The split is by **observation**, not by image — a naive random image split
leaks (val has near-duplicate images of the same observation), which inflated an early
estimate to a fake 87%. The trustworthy figure is ~83%.

| Configuration | top-1 | top-5 | top-10 |
|---|---|---|---|
| Old baselines (ResNet-50 kNN / EfficientNet fine-tune) | ~5–8% | — | — |
| BioCLIP-2 centroid (train-only reference) | 46.7% | 68.3% | 75.4% |
| + TTA + train+val reference | 52.9% | 74.3% | 81.9% |
| **+ soft transductive prototypes (final)** | **53.0%** | **75.4%** | **83.2%** |

---

## 4. What we tried that the data **rejected**

Documented as negative results — each was built and measured, then dropped:

- **Caption text** (BioCLIP-2 text tower on the per-image AI descriptions): top-1 only
  ~3% standalone, and fusing it *degraded* the image model monotonically. The captions
  ("brown cap, fuzzy texture, no visible gills") are too generic to separate 2427
  fine-grained species.
- **Hard pseudo-labelling** (add top-confidence test observations with hard labels to
  the reference): ~flat — the ~25% wrong labels cancel the gains from the right ones.
  The **soft** transductive variant (§2, lever 3) works instead.
- **DINOv3 / k-NN / metadata re-rankers:** non-zero individually, but the ensemble grid
  search drives their weights to ~0 once the train+val reference is in place. The final
  predictor is effectively **BioCLIP-2 transductive centroid alone**.
- **Power transform** on embeddings (a classic few-shot trick): slightly *hurt* — it
  assumes non-negative (ReLU) features, but CLIP embeddings are signed.
- **Full fine-tuning** (CAFormer / EfficientNet): collapsed to ~0% in few-shot.

---

## 5. Files

### Submission
- **`submission_ensemble.csv`** ← **upload this.** Columns `observationId,predictions`,
  999 rows, 10 space-separated `category_id`s each — identical format to
  `data/FungiCLEF25-SAMPLE_SUBMISSION.csv`.

### Code
| File | Role |
|---|---|
| `fungi_features.py` | Extract & cache BioCLIP-2 + DINOv3 embeddings (+ hflip TTA) from 500px images |
| `fungi_captions.py` | BioCLIP-2 text embeddings of captions (tested, unused — kept for transparency) |
| `fungi_train.py` | Metadata encoder, centroid / k-NN / transductive heads, ensemble grid-search, honest CV, submission |
| `fungi_ensemble.ipynb` | Narrated driver — runs the whole pipeline with visualisations end-to-end |

### Artifacts
| File | Contents |
|---|---|
| `features_ensemble/*.npy` | Cached embeddings (re-runs are instant) |
| `ensemble_probs.npz` | Final val/test probabilities + CV metrics |
| `viz_ens_data_overview.png` | Class distribution, seasonality, habitats |
| `viz_ens_experts.png` | Per-expert comparison (BioCLIP-2 dominates) |
| `viz_ens_reference_gain.png` | Accuracy progression across the three levers |
| `viz_ens_diagnostics.png` | Per-class accuracy; accuracy vs #shots-per-class |

---

## 6. How to reproduce

```bash
# environment (system Python 3.12, RTX 5090)
pip install --break-system-packages open_clip_torch timm lightgbm \
    faiss-cpu seaborn nbconvert ipykernel

# 1) extract & cache embeddings (one-time GPU pass; ~minutes)
python3 fungi_features.py

# 2) train heads, ensemble, honest CV, write submission_ensemble.csv
python3 fungi_train.py

#    …or run the narrated notebook end-to-end (same result + figures)
jupyter notebook fungi_ensemble.ipynb
```

Key hyperparameters (in `fungi_train.py`): image resolution `500p`; centroid softmax
temperature `10`; soft transduction `alpha=0.5, iters=3`; ensemble weights grid-searched
on val with a balanced `(top1+top5+top10)/3` objective.

---

## 7. If we wanted to push further

- **Sinkhorn / optimal-transport** label assignment for the transductive step (enforces
  the class-balance prior — typically a stronger transductive method than soft EM).
- **LoRA-finetune BioCLIP-2** with heavy augmentation — parameter-efficient enough to
  avoid the few-shot collapse while still adapting the backbone.
- **5-crop / multi-scale TTA** beyond the single horizontal flip.
