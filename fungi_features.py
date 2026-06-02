"""Feature extraction for FungiCLEF 2025 ensemble pipeline.

Extracts and caches L2-normalized image embeddings from two frozen foundation
backbones:
  * BioCLIP-2  (open_clip, hf-hub:imageomics/bioclip-2) -- 224px, CLIP norm
  * DINOv3-B16 (timm, vit_base_patch16_dinov3.lvd1689m)  -- 256px, ImageNet norm

Both are kept FROZEN. With ~3.2 images/class (few-shot) full fine-tuning
destroys the pretrained features, so we treat the backbones as fixed encoders
and learn only a lightweight head downstream.

Embeddings are cached to features_ensemble/<backbone>_<split>.npy so the heavy
GPU pass runs only once.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # a few dataset images are truncated

DATA_ROOT = "data"
IMG_RES = "500p"  # higher than the 300p baseline -> more fine-grained texture
CACHE_DIR = "features_ensemble"
SPLIT_MAP = {"train": "Train", "val": "Val", "test": "Test"}

os.makedirs(CACHE_DIR, exist_ok=True)


def metadata_csv(split):
    return os.path.join(
        DATA_ROOT, "metadata", "FungiTastic-FewShot",
        f"FungiTastic-FewShot-{SPLIT_MAP[split]}.csv",
    )


def load_split_df(split):
    return pd.read_csv(metadata_csv(split))


class _ImageDataset(Dataset):
    """Returns (transformed_image, row_index) so we keep CSV order."""

    def __init__(self, df, split, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = os.path.join(
            DATA_ROOT, "images", "FungiTastic-FewShot", split, IMG_RES
        )
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fn = self.df.iloc[idx]["filename"]
        img = Image.open(os.path.join(self.img_dir, fn)).convert("RGB")
        return self.transform(img), idx


# --------------------------------------------------------------------------- #
# Backbone loaders
# --------------------------------------------------------------------------- #
def load_bioclip2(device):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip-2"
    )
    model = model.eval().to(device)

    @torch.no_grad()
    def encode(batch):
        return model.encode_image(batch)

    return encode, preprocess


def load_dinov3(device):
    import timm
    model = timm.create_model(
        "vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0
    ).eval().to(device)
    cfg = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**cfg, is_training=False)

    @torch.no_grad()
    def encode(batch):
        return model(batch)

    return encode, preprocess


BACKBONES = {"bioclip2": load_bioclip2, "dinov3": load_dinov3}


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract(backbone, split, device, batch_size=64, num_workers=8):
    """Return (N, D) L2-normalized float32 embeddings, cached on disk."""
    cache = os.path.join(CACHE_DIR, f"{backbone}_{split}_{IMG_RES}.npy")
    if os.path.exists(cache):
        return np.load(cache)

    df = load_split_df(split)
    encode, preprocess = BACKBONES[backbone](device)
    ds = _ImageDataset(df, split, preprocess)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    feats = np.zeros((len(df), 0), dtype=np.float32)
    out = None
    use_amp = device == "cuda"
    for imgs, idxs in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=use_amp):
            emb = encode(imgs).float()
        emb = torch.nn.functional.normalize(emb, dim=1).cpu().numpy()
        if out is None:
            out = np.zeros((len(df), emb.shape[1]), dtype=np.float32)
        out[idxs.numpy()] = emb
    np.save(cache, out)
    return out


def extract_flip(backbone, split, device, batch_size=64, num_workers=8):
    """Same as extract() but with a horizontal flip (test-time augmentation)."""
    from torchvision import transforms
    cache = os.path.join(CACHE_DIR, f"{backbone}_{split}_{IMG_RES}_flip.npy")
    if os.path.exists(cache):
        return np.load(cache)

    df = load_split_df(split)
    encode, preprocess = BACKBONES[backbone](device)
    flip_pre = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=1.0), preprocess]
    )
    ds = _ImageDataset(df, split, flip_pre)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    out = None
    for imgs, idxs in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=(device == "cuda")):
            emb = encode(imgs).float()
        emb = torch.nn.functional.normalize(emb, dim=1).cpu().numpy()
        if out is None:
            out = np.zeros((len(df), emb.shape[1]), dtype=np.float32)
        out[idxs.numpy()] = emb
    np.save(cache, out)
    return out


def tta_embedding(backbone, split, device):
    """L2-normalised mean of original + horizontal-flip embeddings."""
    a = extract(backbone, split, device)
    b = extract_flip(backbone, split, device)
    m = a + b
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-8, None)


def extract_all(device="cuda"):
    res = {}
    for bb in BACKBONES:
        for sp in ("train", "val", "test"):
            arr = extract(bb, sp, device)
            res[(bb, sp)] = arr
            print(f"  {bb:9s} {sp:5s} -> {arr.shape}")
    return res


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev} | resolution: {IMG_RES}")
    extract_all(dev)
    print("Feature extraction complete.")
