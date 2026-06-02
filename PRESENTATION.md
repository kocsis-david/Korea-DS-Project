# FungiCLEF 2025 — Few-Shot Fungal Species Identification
### Presentation / Report (Data Analysis Final Project)

> **Headline result:** Kaggle private **0.79042** / public **0.81858** — above the
> displayed year-old #1 (0.78913). Built on frozen foundation-model embeddings with a
> non-parametric head; **no GPU fine-tuning**, trains in seconds, and every design
> choice is backed by a leakage-free measurement.

This document is structured as the 7 slides to present, with speaker notes. A mapping
to the 7 evaluation criteria is given at the end (§8).

---

## 1. Competition Overview

**FungiCLEF 2025 (FungiTastic–FewShot)** — identify the fungal **species** in a field
photograph, where most species have only a handful of training images.

| | |
|---|---|
| **Goal** | Given a photo (observation), rank the most likely species |
| **Classes** | **2427** fungal species |
| **Train** | 7 819 images · **3.22 images/class on average** |
| **Validation** | 2 285 images (labels public — we exploit this, see §4) |
| **Test** | 1 911 images → **999 observations** (multiple photos per observation) |
| **Output** | Per observation, a ranked **top-10** list of species `category_id`s |
| **Metric** | Recall-style score over the ranked predictions (≈ our top-k recall) |

**Why it's hard — the defining constraint:** this is **extreme few-shot**.
- **707 of 2427 classes (29%) have only ONE training image.**
- **51% have ≤ 2 images.** The long tail dominates.
- Fungi are *fine-grained*: many species differ by subtle cap/gill/spore features.

> **Speaker note:** The whole strategy follows from one number — 3.2 images/class. That
> rules out training a network from scratch and rules out heavy fine-tuning; both need
> far more data per class than we have.

---

## 2. Data Analysis Process

### 2.1 What the dataset contains
- **Images** at 4 resolutions (300/500/720/fullsize px) — we use **500px**.
- **Per-image captions**: AI-generated text descriptions (train/val/**test**).
- **Rich tabular metadata** per observation (see below).
- **Taxonomy labels** (train/val only): phylum (11) → class (32) → order (123) →
  family (364) → genus (1014) → species (2427), plus `poisonous` and IUCN status.

### 2.2 Exploratory findings (drove every later decision)
| Finding | Value | Implication |
|---|---|---|
| Class imbalance | 1–30 imgs/class, median **2** | few-shot regime → prototype methods, not fine-tuning |
| Geographic skew | **91.8% Denmark** (then GR/SE/NO) | location features have limited discriminative range |
| Seasonality | observations peak in **autumn (Sep–Nov)** | `month` is a meaningful signal |
| Toxicity imbalance | non-poisonous : poisonous = **7762 : 57 (1:136)** | a toxicity classifier collapses to all-safe (recall 0) |
| Coordinates | **100%** have GPS | usable, but range is narrow (mostly DK) |

### 2.3 Missing values & outliers (criterion 1)
- **Missingness (train):** substrate 5.1%, family 2.3%, biogeographicalRegion 1.9%,
  district/order 1.4%, landcover 1.3%, elevation 1.2%. No column is critically sparse.
- **Handling — categorical:** every NaN mapped to an explicit `"__nan__"` category in
  one-hot encoding (missingness is itself informative; we don't drop rows).
- **Handling — numeric:** NaNs imputed to the **train mean**, then standardised (so an
  imputed value sits at 0 after scaling and adds no spurious signal).
- **Outliers:** `coorUncert` ranges from 25 m (median) to **50 000 m** (whole-region
  guesses); `elevation` up to 1600 m (a few Greek records vs. flat Denmark, median 14 m).
  We **did not clip** them — they enter only as *standardised auxiliary* features whose
  ensemble weight the model is free to drive to ~0, which it did (§4). This is safer than
  hand-tuned clipping thresholds on a feature that turned out to be weak.
- **Truncated images:** a handful of image files are byte-truncated → enabled
  `PIL.ImageFile.LOAD_TRUNCATED_IMAGES` so extraction never crashes.

### 2.4 Evaluation protocol (so our numbers are trustworthy)
Everything is scored **at the observation level** (mean of an observation's per-image
predictions), matching the competition. To estimate test accuracy *before* submitting,
we use an **observation-disjoint 2-fold CV on val** — a naïve random *image* split leaks
(val has near-duplicate photos of the same observation) and inflated an early estimate to
a fake 87%; the honest figure is ~83%. **Spotting and fixing this leak is a key result.**

---

## 3. Feature Engineering Strategy

We engineered **three feature streams** and tested each empirically:

### 3.1 Image embeddings (the backbone of everything)
Two **frozen** foundation models, embeddings cached once:
- **BioCLIP-2** (`imageomics/bioclip-2`) — a vision-language model trained on the
  **Tree of Life**; in-domain for organism ID. 768-d. *The workhorse.*
- **DINOv3-B/16** (`timm`) — best general self-supervised ViT. 768-d. *Complementary.*

All embeddings are **L2-normalised** (so dot product = cosine similarity).

### 3.2 Test-time augmentation (TTA)
Each image is encoded twice — original + horizontal flip — and the two vectors averaged.
Small, free robustness gain.

### 3.3 Tabular metadata → fusion features (126-d)
- **`month` → sin/cos** (cyclic encoding so Dec is adjacent to Jan).
- **elevation, latitude, longitude →** standardised numeric.
- **habitat, substrate, metaSubstrate, landcover, biogeographicalRegion, countryCode →**
  one-hot (vocabularies fit on the *union* of splits so test-only categories survive).

### 3.4 Caption text (tested, then dropped)
Encoded captions with BioCLIP-2's **text** tower (sentence-split + mean-pool). Measured
result: **top-1 ≈ 3% alone**, and fusing it *degraded* the image model. The captions
("brown cap, fuzzy texture") are too generic to separate 2427 fine species → **removed**.
Reporting the negative is part of the method.

> **Speaker note:** Feature engineering here isn't about inventing many features — it's
> about *measuring* which streams carry signal and letting the data discard the rest.

---

## 4. Model Selection & Reasoning

### 4.1 Why NOT fine-tuning (the failed baseline)
We first tried the obvious thing — fine-tune a strong CNN/transformer (CAFormer,
EfficientNet). With 3 images/class it **collapsed to ~0% (random)** after one epoch:
gradient updates destroy the pretrained features faster than 2427 classes can be learned.
**This failure motivated the whole approach.**

### 4.2 The chosen approach — frozen embeddings + non-parametric head
> **Cosine-centroid (prototype) classifier:** average each class's L2-normalised
> embeddings into a centroid; classify by cosine similarity. Zero parameters → cannot
> overfit, trains instantly, robust in few-shot.

### 4.3 Three accuracy levers (each measured on the honest val CV)
| Lever | Idea | top-10 effect |
|---|---|---|
| **hflip TTA** | average original + flipped embedding | small, free |
| **train + val reference** | fold the **public val labels** into the prototype pool used for test | **+5–6 pts — the biggest win** |
| **soft transductive prototypes** | a few EM steps nudging centroids toward the unlabelled **test** distribution via probability-weighted votes | **+1.3 pts** |

- **Why train+val helps:** val labels are public; only test labels are hidden. Using
  them as reference roughly doubles the shots-per-class available at inference. Legitimate
  and high-impact.
- **Why *soft* transduction (not hard pseudo-labels):** hard pseudo-labelling was ~flat —
  the ~25% wrong labels cancel the right ones. The soft version never commits to a label,
  so errors can't snowball. **A creative, measured improvement** (criterion 7).

### 4.4 What the ensemble grid-search rejected
The weighted ensemble (BioCLIP-2 + DINOv3 + k-NN + metadata, weights tuned on val) drove
DINOv3/k-NN/metadata to **weight ≈ 0** once the strong reference was in place. The final
predictor is effectively **BioCLIP-2 transductive centroid** — simpler *because the data
said so*, not by assumption.

---

## 5. Performance & Results

### 5.1 Ablation (observation-level, leakage-free)
| Configuration | top-1 | top-5 | top-10 |
|---|---|---|---|
| Old baselines (ResNet-50 kNN / EfficientNet FT) | ~5–8% | — | — |
| BioCLIP-2 centroid (train ref) | 46.7% | 68.3% | 75.4% |
| + TTA + train+val reference | 52.9% | 74.3% | 81.9% |
| **+ soft transduction (final)** | **53.0%** | **75.4%** | **83.2%** |

### 5.2 Leaderboard (real, held-out test)
- **Private: 0.79042 · Public: 0.81858** — our honest CV predicted ~0.83, the score
  landed 0.79–0.82. **The estimate matched reality → the model is well-calibrated, not
  overfit.** (An overfit model would have crashed on the private split.)

### 5.3 Diagnostics (figures in repo)
- `viz_ens_reference_gain.png` — accuracy climbing past 80% across the three levers.
- `viz_ens_diagnostics.png` — **accuracy rises monotonically with shots/class** (1-shot
  ≈ 11%, 10+ shots ≈ 80%): the one-shot tail is where the remaining error lives.
- `viz_ens_experts.png` — BioCLIP-2 dominates; DINOv3/metadata weak alone.
- `viz_ens_data_overview.png` — class imbalance, autumn seasonality, top habitats.

---

## 6. Lessons Learned

1. **Match the method to the data regime.** Few-shot ⇒ frozen features + prototypes beat
   fine-tuning by ~6×. The fanciest model (full fine-tune) was the *worst*.
2. **Domain pretraining > model size.** In-domain BioCLIP-2 crushed general DINOv3 on
   fungi (top-1 46.7% vs 17%). *What* a model was trained on mattered more than its size.
3. **Validation design can lie.** A random split leaked via per-observation duplicates and
   reported a fake +5 pts. **Observation-disjoint CV** was essential to trust our numbers.
4. **Negative results are results.** Captions, hard pseudo-labels, DINOv3, metadata, and
   power-transform were all built, measured, and dropped — that's how we *knew* the final
   model was right, instead of guessing.
5. **Exploit what the rules allow.** Folding public val labels into the reference was the
   single biggest lever and is fully legitimate.

---

## 7. Future Plans

- **Sinkhorn / optimal-transport** label assignment for the transductive step — enforces
  the class-balance prior; typically stronger than soft-EM transduction.
- **LoRA fine-tuning of BioCLIP-2** with heavy augmentation — parameter-efficient enough
  to adapt the backbone without the few-shot collapse of full fine-tuning.
- **Multi-scale / 5-crop TTA** beyond a single horizontal flip.
- **Hierarchical taxonomy priors** — re-rank species by genus/family plausibility.
- **Tackle the 1-shot tail directly** — synthetic augmentation or cross-species transfer
  for the 707 single-image classes that cap our ceiling.

---

## 8. Mapping to Evaluation Criteria

| # | Criterion | Where addressed |
|---|---|---|
| 1 | Data exploration & preprocessing (missing values, outliers) | §2.2–2.3 (NaN→category / mean-impute+standardise; outliers kept as down-weightable features; truncated-image handling) |
| 2 | Problem definition & objective | §1 (few-shot goal, metric) + §2.4 (honest evaluation protocol) |
| 3 | Selection & application of methods | §4 (frozen embeddings + prototype, *with reasoning vs the failed fine-tune*) |
| 4 | Result interpretation & insight | §5–§6 (ablation, calibration insight, shots-vs-accuracy, leak discovery) |
| 5 | Slide organisation & visualisation | §1–§7 flow + 4 figures (`viz_ens_*.png`) |
| 6 | Presentation delivery | speaker notes throughout; one narrative thread: "3.2 imgs/class ⇒ everything" |
| 7 | Creativity & problem-solving | leak-free CV; soft (vs hard) transduction; val-as-reference; systematic negative results |

---

## Appendix — Reproducibility
Code: `fungi_features.py` (embeddings+TTA), `fungi_train.py` (heads, ensemble, CV,
submission), `fungi_captions.py` (caption test). Driver: `fungi_ensemble.ipynb`.
Run `python3 fungi_features.py` then `python3 fungi_train.py` → `submission_ensemble.csv`.
Full technical write-up in `doc.md`.
