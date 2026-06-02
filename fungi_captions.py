"""Caption-text embeddings for FungiCLEF 2025 (BioCLIP-2 text encoder).

Every image (train/val/**test**) ships an AI-generated description in
data/captions/<split>/<filename>.json. We encode these with BioCLIP-2's *text*
tower, which shares the embedding space with its image tower.

Captions are multi-paragraph (>> the 77-token CLIP limit), so each caption is
split into sentences, every sentence is encoded, and the sentence embeddings are
mean-pooled into one L2-normalised vector per image. Cached to
features_ensemble/captiontext_<split>.npy.
"""
import os
import re
import json
import numpy as np
import torch

from fungi_features import load_split_df, CACHE_DIR

DATA_ROOT = "data"
_SENT = re.compile(r"(?<=[.!?])\s+")


def _caption_text(split, filename):
    path = os.path.join(DATA_ROOT, "captions", split, filename + ".json")
    try:
        with open(path) as f:
            txt = json.load(f)
    except FileNotFoundError:
        return ""
    return txt if isinstance(txt, str) else str(txt)


def _sentences(text, max_sents=12):
    sents = [s.strip() for s in _SENT.split(text.strip()) if len(s.strip()) > 3]
    return sents[:max_sents] if sents else [text.strip() or "a fungus"]


def extract_caption_features(split, device="cuda", batch_size=256):
    cache = os.path.join(CACHE_DIR, f"captiontext_{split}.npy")
    if os.path.exists(cache):
        return np.load(cache)

    import open_clip
    model = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")[0]
    model = model.eval().to(device)
    tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")

    df = load_split_df(split)
    # flatten all sentences, remember which image each belongs to
    sent_list, owner = [], []
    for i, fn in enumerate(df["filename"].values):
        for s in _sentences(_caption_text(split, fn)):
            sent_list.append(s)
            owner.append(i)
    owner = np.asarray(owner)

    embs = np.zeros((len(sent_list), 768), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(sent_list), batch_size):
            toks = tok(sent_list[i:i + batch_size]).to(device)
            with torch.autocast("cuda", enabled=(device == "cuda")):
                e = model.encode_text(toks).float()
            embs[i:i + batch_size] = torch.nn.functional.normalize(e, dim=1).cpu().numpy()

    # mean-pool sentence embeddings per image, then L2-normalise
    out = np.zeros((len(df), 768), dtype=np.float32)
    counts = np.zeros(len(df))
    np.add.at(out, owner, embs)
    np.add.at(counts, owner, 1.0)
    out /= np.clip(counts[:, None], 1, None)
    out /= np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-8, None)
    np.save(cache, out)
    return out


def extract_all_captions(device="cuda"):
    res = {}
    for sp in ("train", "val", "test"):
        arr = extract_caption_features(sp, device)
        res[sp] = arr
        print(f"  captiontext {sp:5s} -> {arr.shape}")
    return res


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")
    extract_all_captions(dev)
    print("Caption-text extraction complete.")
