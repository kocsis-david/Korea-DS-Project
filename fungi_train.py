"""FungiCLEF 2025 -- frozen-backbone ensemble head + metadata fusion.

Strategy (few-shot: 2427 classes x ~3.2 imgs/class):
  * Backbones (BioCLIP-2, DINOv3) are FROZEN -> we only learn lightweight heads.
  * Metadata (month, habitat, substrate, region, elevation, GPS...) is encoded
    as a tabular vector and fused with the image embeddings.
  * We compare three head families and ensemble them:
       1. cosine-centroid / prototype  (zero training, robust in few-shot)
       2. linear softmax on image embeddings
       3. linear softmax on [image embeddings || metadata]
  * Evaluation is OBSERVATION-LEVEL (matches the competition: one prediction per
    observationID, top-10 species).

Run:  python fungi_train.py
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from fungi_features import (load_split_df, extract_all, tta_embedding,
                            CACHE_DIR, IMG_RES)

NUM_CLASSES = 2427
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RNG = np.random.RandomState(42)
torch.manual_seed(42)


# --------------------------------------------------------------------------- #
# 1. Metadata encoding
# --------------------------------------------------------------------------- #
CAT_COLS = ["habitat", "substrate", "metaSubstrate", "landcover",
            "biogeographicalRegion", "countryCode"]
NUM_COLS = ["elevation", "latitude", "longitude"]


def build_metadata_features(dfs):
    """dfs: dict split->df. Returns dict split->(N,M) float32 metadata matrix.

    Categorical vocabularies are fit on the union of all splits so test-only
    categories don't silently drop. Numeric columns are standardised with
    train statistics; NaNs are imputed to the column mean (-> 0 after scaling).
    """
    train = dfs["train"]

    # categorical vocabularies (union across splits, NaN -> "__nan__")
    vocabs = {}
    for c in CAT_COLS:
        vals = set()
        for df in dfs.values():
            vals |= set(df[c].fillna("__nan__").astype(str).unique())
        vocabs[c] = {v: i for i, v in enumerate(sorted(vals))}

    # numeric standardisation stats from train
    num_stats = {}
    for c in NUM_COLS:
        col = train[c].astype(float)
        num_stats[c] = (col.mean(), col.std() + 1e-6)

    out = {}
    for sp, df in dfs.items():
        blocks = []
        # cyclic month
        m = df["month"].fillna(0).astype(float).values
        blocks.append(np.stack([np.sin(2 * np.pi * m / 12),
                                np.cos(2 * np.pi * m / 12)], axis=1))
        # numeric (standardised, NaN->mean)
        for c in NUM_COLS:
            mu, sd = num_stats[c]
            col = df[c].astype(float).fillna(mu).values
            blocks.append(((col - mu) / sd).reshape(-1, 1))
        # one-hot categoricals
        for c in CAT_COLS:
            voc = vocabs[c]
            oh = np.zeros((len(df), len(voc)), dtype=np.float32)
            idx = df[c].fillna("__nan__").astype(str).map(voc).values
            oh[np.arange(len(df)), idx.astype(int)] = 1.0
            blocks.append(oh)
        out[sp] = np.concatenate(blocks, axis=1).astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
# 2. Heads
# --------------------------------------------------------------------------- #
def cosine_centroid_probs(train_x, train_y, eval_x):
    """Prototype classifier: mean L2-normalised embedding per class -> cosine."""
    tx = F.normalize(torch.tensor(train_x), dim=1)
    centroids = torch.zeros(NUM_CLASSES, tx.shape[1])
    counts = torch.zeros(NUM_CLASSES)
    centroids.index_add_(0, torch.tensor(train_y), tx)
    counts.index_add_(0, torch.tensor(train_y), torch.ones(len(train_y)))
    centroids = centroids / counts.clamp(min=1).unsqueeze(1)
    centroids = F.normalize(centroids, dim=1)
    ex = F.normalize(torch.tensor(eval_x), dim=1)
    sims = ex @ centroids.T            # (N, C) cosine in [-1,1]
    return F.softmax(sims * 10.0, dim=1).numpy()   # temperature-scaled


def transductive_centroid_probs(train_x, train_y, eval_x, alpha=0.5,
                                iters=3, temp=10.0):
    """Soft transductive prototype adaptation.

    Start from count-weighted class centroids, then run a few EM-style steps that
    nudge every centroid toward the *unlabelled* eval distribution using soft
    (probability-weighted) assignments. Unlike hard pseudo-labelling this never
    commits to a wrong label — each eval point contributes a weighted vote — so it
    improves recall without the error-amplification of hard self-training.
    Used transductively at test time (eval_x = the test embeddings)."""
    D = train_x.shape[1]
    C = np.zeros((NUM_CLASSES, D), dtype=np.float32)
    cnt = np.zeros(NUM_CLASSES, dtype=np.float32)
    np.add.at(C, train_y, train_x)
    np.add.at(cnt, train_y, 1.0)
    C = C / np.clip(cnt[:, None], 1, None)
    C = C / np.clip(np.linalg.norm(C, axis=1, keepdims=True), 1e-8, None)

    def _probs(cent):
        s = (eval_x @ cent.T) * temp
        s = s - s.max(1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(1, keepdims=True)

    for _ in range(iters):
        p = _probs(C)                       # (Ne, C) soft assignment
        newC = C * cnt[:, None] + alpha * (p.T @ eval_x)
        C = (newC / np.clip(np.linalg.norm(newC, axis=1, keepdims=True),
                            1e-8, None)).astype(np.float32)
    return _probs(C)


def knn_cosine_probs(train_x, train_y, eval_x, k=5, temp=10.0):
    """Soft k-NN: for each eval point, softmax-weight the k nearest train
    neighbours' cosine similarity into class scores. Often beats a single
    centroid when a class's few shots are multi-modal."""
    tx = F.normalize(torch.tensor(train_x, device=DEVICE), dim=1)
    ty = torch.tensor(train_y, device=DEVICE)
    ex = F.normalize(torch.tensor(eval_x, device=DEVICE), dim=1)
    out = torch.zeros(len(ex), NUM_CLASSES, device=DEVICE)
    bs = 512
    for i in range(0, len(ex), bs):
        sims = ex[i:i + bs] @ tx.T                  # (b, Ntrain)
        topv, topi = sims.topk(k, dim=1)
        w = F.softmax(topv * temp, dim=1)           # (b, k)
        cls = ty[topi]                              # (b, k)
        out[i:i + bs].scatter_add_(1, cls, w)
    return (out / out.sum(1, keepdim=True).clamp(min=1e-8)).cpu().numpy()


def train_linear_head(train_x, train_y, eval_x, epochs=60, lr=1e-3,
                      wd=1e-3, class_weight=None):
    """GPU linear softmax head. Returns eval probabilities (N, C)."""
    Xtr = torch.tensor(train_x, device=DEVICE)
    ytr = torch.tensor(train_y, device=DEVICE, dtype=torch.long)
    Xev = torch.tensor(eval_x, device=DEVICE)

    head = nn.Linear(Xtr.shape[1], NUM_CLASSES).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    cw = None if class_weight is None else torch.tensor(
        class_weight, device=DEVICE, dtype=torch.float)

    bs = 1024
    n = len(Xtr)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        head.train()
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = F.cross_entropy(head(Xtr[b]), ytr[b], weight=cw)
            loss.backward()
            opt.step()
        sched.step()
    head.eval()
    with torch.no_grad():
        probs = F.softmax(head(Xev), dim=1).cpu().numpy()
    return probs


# --------------------------------------------------------------------------- #
# 3. Observation-level evaluation
# --------------------------------------------------------------------------- #
def observation_topk(probs, df, ks=(1, 5, 10), labels=None):
    """Aggregate per-image probs by observationID (mean), report top-k recall.

    Returns (metrics_dict, obs_ids, obs_topk_indices).
    """
    obs_ids = df["observationID"].values
    uniq = pd.unique(obs_ids)
    obs_probs = np.zeros((len(uniq), probs.shape[1]), dtype=np.float32)
    pos = {o: i for i, o in enumerate(uniq)}
    counts = np.zeros(len(uniq))
    for i, o in enumerate(obs_ids):
        obs_probs[pos[o]] += probs[i]
        counts[pos[o]] += 1
    obs_probs /= counts[:, None]

    order = np.argsort(-obs_probs, axis=1)
    topk_idx = order[:, :max(ks)]

    metrics = {}
    if labels is not None:
        # one label per observation
        obs_label = np.zeros(len(uniq), dtype=int)
        for i, o in enumerate(obs_ids):
            obs_label[pos[o]] = labels[i]
        for k in ks:
            hit = (topk_idx[:, :k] == obs_label[:, None]).any(axis=1).mean()
            metrics[f"top{k}"] = float(hit)
    return metrics, uniq, topk_idx


def fmt(m):
    return "  ".join(f"{k}={v:.4f}" for k, v in m.items())


# --------------------------------------------------------------------------- #
# 4. Main
# --------------------------------------------------------------------------- #
def build_experts(ref, ref_y, eval_set, class_weight):
    """Build the four experts for a given reference (prototype/training) pool.

    ref / eval_set: dicts with keys 'bioclip2', 'dinov3', 'meta'.
    Returns name -> (N_eval, C) probability matrix.
    """
    ex = {}
    ex["cent_bioclip2"] = cosine_centroid_probs(ref["bioclip2"], ref_y, eval_set["bioclip2"])
    ex["cent_dinov3"] = cosine_centroid_probs(ref["dinov3"], ref_y, eval_set["dinov3"])
    ex["knn_bioclip2"] = knn_cosine_probs(ref["bioclip2"], ref_y, eval_set["bioclip2"])
    ex["meta"] = train_linear_head(ref["meta"], ref_y, eval_set["meta"], class_weight=class_weight)
    return ex


def geo_combine(experts, weights):
    """Weighted geometric mean of expert probabilities, returned in PROBABILITY
    space (per-row softmax of the weighted log-sum) so that the downstream
    per-observation aggregation is an arithmetic mean of probabilities."""
    eps = 1e-8
    logp = np.zeros_like(experts["cent_bioclip2"])
    for name, w in weights.items():
        if w:
            logp = logp + w * np.log(experts[name] + eps)
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def main():
    print(f"Device: {DEVICE} | features @ {IMG_RES} (+ hflip TTA)")
    dfs = {sp: load_split_df(sp) for sp in ("train", "val", "test")}
    y_train = dfs["train"]["category_id"].astype(int).values
    y_val = dfs["val"]["category_id"].astype(int).values

    freq = np.bincount(y_train, minlength=NUM_CLASSES).astype(float)
    class_weight = np.where(freq > 0, 1.0 / np.sqrt(freq), 0.0)
    class_weight = class_weight / class_weight[freq > 0].mean()

    print("Loading cached embeddings (TTA = original + hflip)...")
    img = {sp: {bb: tta_embedding(bb, sp, DEVICE) for bb in ("bioclip2", "dinov3")}
           for sp in ("train", "val", "test")}
    for sp in ("train", "val", "test"):
        print(f"  {sp:5s}: bioclip {img[sp]['bioclip2'].shape}  dino {img[sp]['dinov3'].shape}")

    print("Encoding metadata...")
    meta = build_metadata_features(dfs)
    print(f"  metadata dim = {meta['train'].shape[1]}")

    def pack(sp):
        return {"bioclip2": img[sp]["bioclip2"], "dinov3": img[sp]["dinov3"], "meta": meta[sp]}

    # ------------------------------------------------------------------ #
    # (A) VAL tuning: reference = TRAIN only (honest dev setup).
    # ------------------------------------------------------------------ #
    print("\n=== Experts on VAL (reference = train only) ===")
    val_ex = build_experts(pack("train"), y_train, pack("val"), class_weight)
    for name, p in val_ex.items():
        m, *_ = observation_topk(p, dfs["val"], labels=y_val)
        print(f"  {name:16s}: {fmt(m)}")

    def score(m):
        return (m["top1"] + m["top5"] + m["top10"]) / 3.0

    print("\n=== Grid search ensemble weights on VAL (balanced top1/5/10) ===")
    best = {"_s": -1}
    for w_knn in (0.0, 0.3, 0.5, 1.0):
        for w_dn in (0.0, 0.1, 0.2):
            for w_meta in (0.0, 0.1, 0.2, 0.3, 0.5):
                w = {"cent_bioclip2": 1.0, "knn_bioclip2": w_knn,
                     "cent_dinov3": w_dn, "meta": w_meta}
                m, *_ = observation_topk(geo_combine(val_ex, w), dfs["val"], labels=y_val)
                if score(m) > best["_s"]:
                    best = {**m, "_s": score(m), "w": w}
    bw = best["w"]
    active = {k: v for k, v in bw.items() if v}
    print(f"  best weights: {active}")
    print(f"  VAL (train-only ref): top1={best['top1']:.4f}  "
          f"top5={best['top5']:.4f}  top10={best['top10']:.4f}")

    # ------------------------------------------------------------------ #
    # (B) Honest TEST estimate: observation-disjoint 2-fold on val, where
    #     each eval fold uses train + the *other* val fold as reference.
    #     This mimics the real test (train+val reference, unseen observations).
    # ------------------------------------------------------------------ #
    print("\n=== Honest test estimate (obs-disjoint val CV, ref = train+val-fold) ===")
    obs = dfs["val"]["observationID"].values
    uniq = pd.unique(obs)
    rng = np.random.RandomState(0)
    order = rng.permutation(len(uniq))
    setA = set(uniq[order[:len(uniq) // 2]])
    A = np.array([i for i, o in enumerate(obs) if o in setA])
    B = np.array([i for i, o in enumerate(obs) if o not in setA])
    cv = {"ens": {"top1": [], "top5": [], "top10": []},
          "trans": {"top1": [], "top5": [], "top10": []}}
    for ev_idx, rf_idx in [(A, B), (B, A)]:
        ref = {k: np.concatenate([pack("train")[k], pack("val")[k][rf_idx]]) for k in pack("train")}
        ref_y = np.concatenate([y_train, y_val[rf_idx]])
        ev_set = {k: pack("val")[k][ev_idx] for k in pack("val")}
        ex = build_experts(ref, ref_y, ev_set, class_weight)
        m, *_ = observation_topk(geo_combine(ex, bw), dfs["val"].iloc[ev_idx], labels=y_val[ev_idx])
        for k in cv["ens"]:
            cv["ens"][k].append(m[k])
        # transductive refinement on the strong backbone
        tp = transductive_centroid_probs(ref["bioclip2"], ref_y, ev_set["bioclip2"])
        mt, *_ = observation_topk(tp, dfs["val"].iloc[ev_idx], labels=y_val[ev_idx])
        for k in cv["trans"]:
            cv["trans"][k].append(mt[k])
    e, t = cv["ens"], cv["trans"]
    print(f"  ensemble        : top1={np.mean(e['top1']):.4f}  "
          f"top5={np.mean(e['top5']):.4f}  top10={np.mean(e['top10']):.4f}")
    print(f"  + transductive  : top1={np.mean(t['top1']):.4f}  "
          f"top5={np.mean(t['top5']):.4f}  top10={np.mean(t['top10']):.4f}  "
          f"(real test uses ALL of val -> expect slightly higher)")

    # ------------------------------------------------------------------ #
    # (C) Final TEST predictions: reference = train + ALL of val, with
    #     soft transductive centroid refinement on the test embeddings.
    # ------------------------------------------------------------------ #
    print("\n=== Final test inference (ref = train + ALL val, transductive) ===")
    ref_all = {k: np.concatenate([pack("train")[k], pack("val")[k]]) for k in pack("train")}
    ref_all_y = np.concatenate([y_train, y_val])
    test_probs = transductive_centroid_probs(
        ref_all["bioclip2"], ref_all_y, pack("test")["bioclip2"])

    _, obs_ids, topk_idx = observation_topk(test_probs, dfs["test"], ks=(10,))
    sub = pd.DataFrame({
        "observationId": obs_ids,
        "predictions": [" ".join(map(str, row)) for row in topk_idx[:, :10]],
    })
    sample = pd.read_csv("data/FungiCLEF25-SAMPLE_SUBMISSION.csv")
    sub = sample[["observationId"]].merge(sub, on="observationId", how="left")
    sub.to_csv("submission_ensemble.csv", index=False)
    print(f"Wrote submission_ensemble.csv  shape={sub.shape}")
    print(sub.head(3).to_string())

    np.savez_compressed("ensemble_probs.npz",
                        val=geo_combine(val_ex, bw), test=test_probs,
                        y_val=y_val, weights=str(bw),
                        cv_top1=np.mean(t["top1"]), cv_top10=np.mean(t["top10"]))
    print("Saved ensemble_probs.npz")


if __name__ == "__main__":
    main()
