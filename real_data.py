from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split

# Make PIL robust to slightly truncated JPEGs
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from torchvision import transforms as T
except Exception:
    T = None  # We can still run without torchvision; just require tensors upstream.


# -------------------------------
# Helpers
# -------------------------------

def _norm_path(p: str) -> str:
    """Normalize Windows/Unix paths without altering drive letters."""
    return os.path.normpath(p)

def _exists(p: str) -> bool:
    try:
        return os.path.isfile(p)
    except Exception:
        return False

def _is_obj_entry(x: Any) -> bool:
    return isinstance(x, dict) and "path" in x

def _as_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    raise ValueError("Manifest JSON must be a list of strings or objects.")


# -------------------------------
# Manifest loading / filtering
# -------------------------------

@dataclass
class Entry:
    path: str
    meta: Dict[str, Any]

def load_manifest(
    manifest_path: str,
    require_exists: bool = True,
    filters: Optional[Dict[str, Union[str, Sequence[str]]]] = None,
) -> List[Entry]:
    """
    Load a BDD100K-style manifest of image paths and optionally filter by metadata.

    Args:
      manifest_path: path to bdd100k_manifest.json
      require_exists: skip entries whose files are missing
      filters: dict of key -> value or key -> [values] to keep (match-all)

    Returns:
      List[Entry]
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = _as_list(raw)
    out: List[Entry] = []

    def passes(meta: Dict[str, Any]) -> bool:
        if not filters:
            return True
        for k, want in filters.items():
            if k not in meta:
                return False
            v = meta[k]
            if isinstance(want, (list, tuple, set)):
                if v not in want:
                    return False
            else:
                if v != want:
                    return False
        return True

    for i, item in enumerate(data):
        if isinstance(item, str):
            path = _norm_path(item)
            meta = {}
        elif _is_obj_entry(item):
            path = _norm_path(str(item["path"]))
            meta = {k: v for k, v in item.items() if k != "path"}
        else:
            raise ValueError(
                f"Manifest entry at index {i} must be a string path or an object with 'path'."
            )

        if require_exists and not _exists(path):
            continue
        if passes(meta):
            out.append(Entry(path=path, meta=meta))

    if not out:
        raise ValueError("No entries loaded (after filters / existence checks).")
    return out


# -------------------------------
# Dataset
# -------------------------------

class RealImageDataset(Dataset):
    """
    Minimal dataset to read RGB images listed in a manifest.

    Returns:
      image: FloatTensor [C,H,W] in [0,1] (if default transform used)
      meta:  dict with 'path', 'index', and any extra manifest keys
    """

    def __init__(
        self,
        entries: List[Entry],
        transform: Optional[Any] = None,
        skip_errors: bool = True,
        convert_mode: str = "RGB",
        img_size: Optional[int] = None,
    ):
        self.entries = entries
        self.convert_mode = convert_mode
        self.skip_errors = skip_errors
        self.img_size = img_size

        # Default transforms: ToTensor only (preserve native size) unless img_size provided
        if transform is not None:
            self.transform = transform
        else:
            if T is None:
                raise ImportError(
                    "torchvision is not available. Please install torchvision "
                    "or pass a custom transform that converts PIL->Tensor."
                )
            ops = []
            if img_size:
                # Resize shortest side to img_size and center-crop to square
                ops.extend([T.Resize(img_size), T.CenterCrop(img_size)])
            ops.append(T.ToTensor())
            self.transform = T.Compose(ops)

        # Precompute valid indices (existence already handled in load_manifest)
        self._valid_idx = list(range(len(self.entries)))
        if not self._valid_idx:
            raise ValueError("No valid entries in dataset.")

    def __len__(self) -> int:
        return len(self._valid_idx)

    def _load_pil(self, path: str) -> Image.Image:
        img = Image.open(path)
        if self.convert_mode:
            img = img.convert(self.convert_mode)
        return img

    def __getitem__(self, i: int):
        """
        Return (image_tensor [C,H,W] in [0,1], meta_dict).
        Supports entries either dicts {"path": ...} or objects with .path.
        """
        e = self.entries[i]
        if isinstance(e, dict) and "path" in e:
            path = os.path.normpath(str(e["path"]))
            extra_meta = {k: v for k, v in e.items() if k != "path"}
        elif hasattr(e, "path"):
            path = os.path.normpath(str(e.path))
            extra_meta = dict(getattr(e, "meta", {}) or {})
        else:
            raise TypeError(f"Unsupported entry at index {i}: {type(e)}")

        try:
            pil = self._load_pil(path)
            img = self.transform(pil)
        except Exception as ex:
            if self.skip_errors:
                c = 3 if self.convert_mode.upper() == "RGB" else 1
                h = w = self.img_size if self.img_size else 224
                img = torch.zeros((c, h, w), dtype=torch.float32)
                extra_meta = {**extra_meta, "error": str(ex)}
            else:
                raise

        meta = {"path": path, "index": i}
        meta.update(extra_meta)
        return img, meta

# -------------------------------
# Split + DataLoader factories
# -------------------------------

def split_entries(
    entries: List[Entry],
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Entry], List[Entry], List[Entry]]:
    assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1.0"
    n = len(entries)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train)
    n_val = int(n * val)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    def pick(idxs: Iterable[int]) -> List[Entry]:
        return [entries[int(k)] for k in idxs]

    return pick(train_idx), pick(val_idx), pick(test_idx)

def make_loader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

def get_datasets(
    manifest_path: str,
    img_size: Optional[int] = None,
    filters: Optional[Dict[str, Union[str, Sequence[str]]]] = None,
    split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    convert_mode: str = "RGB",
) -> Dict[str, RealImageDataset]:
    entries = load_manifest(manifest_path, require_exists=True, filters=filters)
    tr, va, te = split_entries(entries, train=split[0], val=split[1], test=split[2], seed=seed)
    ds_train = RealImageDataset(tr, img_size=img_size, convert_mode=convert_mode)
    ds_val   = RealImageDataset(va, img_size=img_size, convert_mode=convert_mode, skip_errors=False)
    ds_test  = RealImageDataset(te, img_size=img_size, convert_mode=convert_mode, skip_errors=False)
    return {"train": ds_train, "val": ds_val, "test": ds_test}

def get_dataloaders(
    manifest_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    img_size: Optional[int] = None,
    filters: Optional[Dict[str, Union[str, Sequence[str]]]] = None,
    split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    convert_mode: str = "RGB",
) -> Dict[str, DataLoader]:
    dsets = get_datasets(
        manifest_path=manifest_path,
        img_size=img_size,
        filters=filters,
        split=split,
        seed=seed,
        convert_mode=convert_mode,
    )
    return {
        "train": make_loader(dsets["train"], batch_size=batch_size, shuffle=True,  num_workers=num_workers),
        "val":   make_loader(dsets["val"],   batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test":  make_loader(dsets["test"],  batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


# -------------------------------
# CLI smoke test
# -------------------------------

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, required=True, help="Path to bdd100k_manifest.json")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--img-size", type=int, default=None, help="If set, resize+center-crop to square")
    p.add_argument("--sample", type=int, default=1, help="How many batches to iterate for the smoke test")
    args = p.parse_args()

    loaders = get_dataloaders(
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )
    for split in ["train", "val", "test"]:
        it = iter(loaders[split])
        for _ in range(args.sample):
            imgs, meta = next(it)
            print(f"[{split}] batch: {tuple(imgs.shape)} | example path: {meta['path'][0]}")

if __name__ == "__main__":
    _cli()
