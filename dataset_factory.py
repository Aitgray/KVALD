# dataset_factory.py — optimized with online self‑supervised glare synthesis
from __future__ import annotations
import json
import os
import math
import random
import numpy as np
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset

# Optional fast image path (preferred when available in repo)
try:
    from real_data import RealImageDataset, load_manifest  # type: ignore
except Exception:
    RealImageDataset, load_manifest = None, None

try:
    from real_data import RealGlareDataset  # if you already implemented a real+synthetic pipeline
except Exception:
    RealGlareDataset = None
    # raise ImportWarning("real_data.RealGlareDataset not available. Set data.source to 'glare_synth_online' or implement RealGlareDataset.") # testing only

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _root_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.isabs(config_path):
        config_path = os.path.join(_root_dir(), config_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_root_dir(), p)


# ------------------------------------------------------------
# Online Glare Synthesis (self‑supervised labels)
# ------------------------------------------------------------
# This dataset composes randomized, physically‑inspired glare layers onto clean frames
# and returns (grayscale_frame, glare_mask). Mask is derived from the synthetic alpha
# used in compositing — no manual labels required.

class GlareSynthOnline(Dataset):
    """Online synthetic glare generation over clean frames.

    Returns (frame_tensor, mask_tensor) where both are [1,H,W] in [0,1].
    The mask is a binary float tensor derived from the synthesized glare alpha.

    Config (data.synth):
      - base_seed: int (default 1337) — deterministic per index
      - severity_range: [lo, hi] in {0..5} controls size/alpha of glare
      - alpha_range: [min, max] of base alpha (e.g., [0.25, 0.95])
      - n_spots: number of Gaussian hotspots (e.g., 1..4)
      - n_streaks: number of radial streaks (e.g., 4..16)
      - streak_decay: radial decay factor (float)
      - mask_threshold: threshold on alpha to form binary mask (e.g., 0.15)
      - grayscale: bool (default True)
      - color_jitter_prob: float (0..1) — small brightness jitter after compose
      - refine_dir: optional path to a folder of pre‑refined glare layers (npz/pt)
    """

    def __init__(
        self,
        entries: List[dict],
        img_size: int | None,
        synth_cfg: Dict[str, Any],
    ):
        self.entries = entries
        self.img_size = img_size
        self.synth = {
            "base_seed": synth_cfg.get("base_seed", 1337),
            "severity_range": synth_cfg.get("severity_range", [1, 4]),
            "alpha_range": synth_cfg.get("alpha_range", [0.25, 0.95]),
            "n_spots": synth_cfg.get("n_spots", [1, 3]),
            "n_streaks": synth_cfg.get("n_streaks", [6, 12]),
            "streak_decay": synth_cfg.get("streak_decay", 2.25),
            "mask_threshold": synth_cfg.get("mask_threshold", 0.15),
            "grayscale": synth_cfg.get("grayscale", True),
            "color_jitter_prob": synth_cfg.get("color_jitter_prob", 0.15),
            "refine_dir": synth_cfg.get("refine_dir", None),
            "fraction": synth_cfg.get("fraction", 1.0),
        }

        # Optional sprite-based glare (e.g., Flare-R Compound_Flare)
        self.sprites: list[torch.Tensor] = []
        sprite_dir = synth_cfg.get("sprite_dir", None)
        self.sprite_prob = float(synth_cfg.get("sprite_prob", 0.0))
        if sprite_dir is not None:
            sprite_dir = _resolve_path(sprite_dir)
            exts = {".png", ".jpg", ".jpeg", ".bmp"}
            try:
                from PIL import Image  # local import
                for fn in os.listdir(sprite_dir):
                    if os.path.splitext(fn)[1].lower() not in exts:
                        continue
                    path = os.path.join(sprite_dir, fn)
                    img = Image.open(path).convert("L")  # grayscale mask-ish
                    arr = np.array(img, dtype=np.float32) / 255.0  # [H,W] in [0,1]
                    t = torch.from_numpy(arr)  # [H,W]
                    if t.max() > 0:
                        t = t / t.max()
                    self.sprites.append(t)
                if self.sprites:
                    print(f"Loaded {len(self.sprites)} glare sprites from {sprite_dir}")
                else:
                    print(f"[GlareSynthOnline] No valid sprites found in {sprite_dir}")
            except Exception as e:
                print(f"[GlareSynthOnline] Failed to load sprites from {sprite_dir}: {e}")
                self.sprites = []
                self.sprite_prob = 0.0

        # If available, use RealImageDataset for fast, consistent decode/resize
        if RealImageDataset is not None and load_manifest is not None:
            self.base = RealImageDataset(entries, img_size=img_size)
        else:
            # Minimal fallback loader (PIL) — avoids tight coupling
            from PIL import Image  # lazy import
            self.PIL = Image
            self.base = None  # use _load_image_fallback

    def __len__(self) -> int:
        return len(self.entries)

    # ------------------------ image IO ------------------------
    def _load_image_fallback(self, path: str):
        # Returns [3,H,W] float in [0,1]
        img = self.PIL.open(path).convert("RGB")
        if self.img_size is not None:
            img = img.resize((self.img_size, self.img_size), self.PIL.BILINEAR)
        t = torch.from_numpy(np.array(img)).float() / 255.0  # [H,W,3]
        # luminance to [1,H,W] - commented out as per user request
        # r, g, b = t[..., 0], t[..., 1], t[..., 2]
        # y = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # return y.unsqueeze(0).permute(0, 1, 2)  # [1,H,W]
        return t.permute(2, 0, 1)

    def _get_clean_frame(self, idx: int) -> torch.Tensor:
        if self.base is not None:
            frame, _ = self.base[idx]  # RealImageDataset returns [C,H,W] in [0,1]
            # The following block converts RGB to grayscale. Commenting it out as requested.
            # if frame.dim() == 3 and frame.size(0) == 3:
            #     r, g, b = frame[0], frame[1], frame[2]
            #     frame = (0.2989 * r + 0.5870 * g + 0.1140 * b).unsqueeze(0)
            if frame.dim() == 2:
                frame = frame.unsqueeze(0)
            # If single channel, repeat to 3 channels to ensure consistency
            if frame.size(0) == 1:
                frame = frame.repeat(3, 1, 1)
            return frame
        else:
            return self._load_image_fallback(self.entries[idx]["path"])  # type: ignore

    # --------------------- glare primitives -------------------
    @staticmethod
    def _make_mesh(H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        ys = torch.linspace(-1.0, 1.0, H, device=device).unsqueeze(1).expand(H, W)
        xs = torch.linspace(-1.0, 1.0, W, device=device).unsqueeze(0).expand(H, W)
        return xs, ys

    @staticmethod
    def _gaussian_2d(xs, ys, mux, muy, sx, sy):
        return torch.exp(-(((xs - mux) ** 2) / (2 * sx * sx) + ((ys - muy) ** 2) / (2 * sy * sy)))

    @staticmethod
    def _radial_streaks(xs, ys, cx, cy, n_streaks, decay):
        # Create n evenly‑spaced angles; streak intensity decays with angle distance
        ang = torch.atan2(ys - cy, xs - cx)  # [-pi, pi]
        base = torch.zeros_like(xs)
        for k in range(n_streaks):
            theta = k * (math.pi * 2.0 / n_streaks)
            d = torch.cos(ang - theta).clamp(min=0)
            base = torch.maximum(base, d ** decay)
        # normalize to [0,1]
        return base / (base.max() + 1e-6)

    def _severity_to_params(self, severity: int, H: int, W: int):
        # Map severity (0..5) to spot size and base alpha multiplier
        size = {
            0: 0.03, 1: 0.05, 2: 0.08, 3: 0.12, 4: 0.17, 5: 0.23,
        }[int(max(0, min(5, severity)))]
        rad = size * max(H, W)
        alpha_mul = 0.5 + 0.08 * severity
        return rad, alpha_mul

    def _synthesize_glare_alpha(self, H: int, W: int, rng: random.Random) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        xs, ys = self._make_mesh(H, W, device)

        sev_lo, sev_hi = self.synth["severity_range"]
        severity = rng.randint(int(sev_lo), int(sev_hi))
        rad, alpha_mul = self._severity_to_params(severity, H, W)

        # Center placement (biased towards image center but anywhere in frame)
        cx = rng.uniform(-0.25, 0.25)
        cy = rng.uniform(-0.25, 0.25)

        # Hotspots
        nspots_lo, nspots_hi = self.synth["n_spots"]
        n_spots = rng.randint(int(nspots_lo), int(nspots_hi))
        spot = torch.zeros((H, W), device=device)
        for _ in range(n_spots):
            mux = cx + rng.uniform(-0.35, 0.35)
            muy = cy + rng.uniform(-0.35, 0.35)
            sx = (rad / max(W, 1)) * rng.uniform(0.6, 1.4)
            sy = (rad / max(H, 1)) * rng.uniform(0.6, 1.4)
            spot = torch.maximum(spot, self._gaussian_2d(xs, ys, mux, muy, sx, sy))

        # Starburst streaks
        nst_lo, nst_hi = self.synth["n_streaks"]
        n_streaks = rng.randint(int(nst_lo), int(nst_hi))
        decay = float(self.synth["streak_decay"])  # angular falloff
        streaks = self._radial_streaks(xs, ys, cx, cy, n_streaks, decay)

        # Additional starburst pattern
        n_streaks2 = rng.randint(2, 20)
        streaks2 = self._radial_streaks(xs, ys, cx, cy, n_streaks2, decay=1.5)

        # Combine and scale into alpha
        base = (0.5 * spot + 0.25 * streaks + 0.25 * streaks2)
        alpha_lo, alpha_hi = self.synth["alpha_range"]
        alpha = base * rng.uniform(alpha_lo, alpha_hi) * alpha_mul
        alpha = alpha.clamp(0.0, 1.0)

        # Ensure glare coverage is <= 30%
        thr = float(self.synth["mask_threshold"])
        if H > 0 and W > 0:
            coverage = (alpha > thr).sum().float() / (H * W)
            if coverage > 0.30:
                low = 0.0
                high = 1.0
                # 10 iterations of binary search for a scaling factor
                for _ in range(10):
                    mid = (low + high) / 2.0
                    if (alpha * mid > thr).sum().float() / (H * W) > 0.30:
                        high = mid
                    else:
                        low = mid
                alpha = alpha * low

        return alpha
    
    def _sprite_alpha(self, H: int, W: int, rng: random.Random) -> torch.Tensor:
        """Sample a precomputed glare sprite and place it into the frame.

        Returns alpha in [0,1] on CPU of shape [H,W].
        """
        if not self.sprites:
            # fallback to procedural if somehow called with no sprites
            return self._synthesize_glare_alpha(H, W, rng)

        # pick a sprite
        idx = rng.randrange(len(self.sprites))
        spr = self.sprites[idx]  # [h,w] in [0,1], CPU

        h, w = spr.shape
        if h == 0 or w == 0:
            return self._synthesize_glare_alpha(H, W, rng)

        # random scale
        scale = rng.uniform(0.5, 1.5)
        new_h = max(8, min(H, int(h * scale)))
        new_w = max(8, min(W, int(w * scale)))

        spr_img = spr.unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
        spr_img = torch.nn.functional.interpolate(
            spr_img, size=(new_h, new_w), mode="bilinear", align_corners=False
        )[0, 0]  # [new_h, new_w]

        # normalize again for safety
        if spr_img.max() > 0:
            spr_img = spr_img / spr_img.max()

        # random placement (biased toward center)
        cy = rng.randint(int(0.25 * H), int(0.75 * H))
        cx = rng.randint(int(0.25 * W), int(0.75 * W))
        top = max(0, cy - new_h // 2)
        left = max(0, cx - new_w // 2)
        bottom = min(H, top + new_h)
        right = min(W, left + new_w)

        alpha = torch.zeros((H, W), dtype=torch.float32)
        alpha[top:bottom, left:right] = spr_img[0 : bottom - top, 0 : right - left]

        return alpha.clamp(0.0, 1.0)


    # ------------------------- compose ------------------------
    @staticmethod
    def _compose(src: torch.Tensor, alpha: torch.Tensor, glare_color: torch.Tensor) -> torch.Tensor:
        # src: [3,H,W], alpha: [H,W], glare_color: [3]
        # reshape glare_color to be broadcastable
        glare_color = glare_color.view(3, 1, 1)
        # Alpha blending
        return src * (1.0 - alpha) + glare_color * alpha

    def __getitem__(self, idx: int):
        # Deterministic RNG per index for reproducible training/debug
        rng = random.Random(self.synth["base_seed"] * 1000003 + idx)

        frame = self._get_clean_frame(idx)  # [C,H,W] in [0,1], must be on CPU
        frame = frame.contiguous()
        _, H, W = frame.shape

        # Make sure frame is on CPU
        if frame.device.type != "cpu":
            frame = frame.cpu()

        if rng.random() > self.synth["fraction"]:
            # Return clean frame with zero mask (CPU)
            mask = torch.zeros((1, H, W), dtype=frame.dtype)  # no device=...
            return frame, frame.clone(), mask

        # Synthesize alpha on CPU: either procedural or sprite-based
        use_sprite = self.sprites and (rng.random() < self.sprite_prob)
        if use_sprite:
            alpha = self._sprite_alpha(H, W, rng)  # [H,W]
        else:
            alpha = self._synthesize_glare_alpha(H, W, rng)  # [H,W]


        # Ensure alpha is CPU + same dtype
        if alpha.device.type != "cpu":
            alpha = alpha.cpu()
        alpha = alpha.to(dtype=frame.dtype)

        # 5% chance of red glare, 95% white (CPU tensor)
        if rng.random() < 0.05:
            glare_color = torch.tensor([1.0, 0.0, 0.0], dtype=frame.dtype)
        else:
            glare_color = torch.tensor([1.0, 1.0, 1.0], dtype=frame.dtype)

        # --------------------------
        # Thresholded alpha for BOTH
        # composition and mask
        # --------------------------
        thr = float(self.synth["mask_threshold"])

        # Hard-mask alpha: no glare outside GT region
        alpha_mask = (alpha > thr).to(frame.dtype)      # [H,W] in {0,1}
        # Optional: rescale inside region so it's still smooth:
        # alpha_comp = ((alpha - thr).clamp(min=0) / (1.0 - thr + 1e-6)) * alpha_mask
        alpha_comp = alpha * alpha_mask                 # simplest version

        # Compose on CPU using masked alpha
        out = self._compose(frame, alpha_comp, glare_color)
        if rng.random() < float(self.synth["color_jitter_prob"]):
            jitter = 1.0 + rng.uniform(-0.08, 0.08)
            out = (out * jitter).clamp(0.0, 1.0)

        # GT mask derived from the SAME gate
        mask = alpha_mask.unsqueeze(0)  # [1,H,W]

        return frame.contiguous(), out.contiguous(), mask.contiguous()


# ------------------------------------------------------------
# Fallback: images with zero masks (keeps pipeline running)
# ------------------------------------------------------------
class _ImagesWithZeroMask(Dataset):
    def __init__(self, entries: List[dict], img_size: int | None = None):
        if RealImageDataset is None:
            raise ImportError("RealImageDataset not available. Provide real_data.RealImageDataset.")
        self.base = RealImageDataset(entries, img_size=img_size)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        frame, _ = self.base[i]  # frame: [C,H,W] in [0,1]
        # The following block converts RGB to grayscale. Commenting it out as requested.
        # if frame.dim() == 3 and frame.size(0) == 3:
        #     r, g, b = frame[0], frame[1], frame[2]
        #     frame = 0.2989 * r + 0.5870 * g + 0.1140 * b
        #     frame = frame.unsqueeze(0)
        if frame.dim() == 2:
            frame = frame.unsqueeze(0)
        if frame.size(0) == 1:
            frame = frame.repeat(3,1,1)
        _, H, W = frame.shape
        mask = torch.zeros((1, H, W), dtype=frame.dtype)
        return frame, frame, mask
    
class Flare7KSegDataset(Dataset):
    """
    Minimal wrapper for Flare7K synthetic segmentation.

    Expects a directory structure like:

        .../synthetic/input/   # flare-corrupted images, e.g. input_000087.png
        .../synthetic/mask/    # masks, e.g. mask_000087.png

    Returns (clean_frame, frame_for_model, mask) so it matches the
    (clean, glare, mask) signature that train.py already expects:
        for _, frames, masks in loader: ...
    """
    def __init__(self, image_dir: str, mask_dir: str, img_size: int | None = None):
        from PIL import Image  # keep module off self to avoid pickling issues

        self.Image = Image
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.img_size  = img_size

        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        mask_files = []
        for fn in os.listdir(mask_dir):
            if os.path.splitext(fn)[1].lower() in exts:
                mask_files.append(fn)
        if not mask_files:
            raise RuntimeError(f"No mask files found under {mask_dir}")

        # Build (img_path, mask_path) pairs with filename mapping
        self.samples: list[tuple[str, str]] = []
        for mname in sorted(mask_files):
            mbase, mext = os.path.splitext(mname)

            # Strip "mask_" prefix if present
            core = mbase
            if core.startswith("mask_"):
                core = core[len("mask_"):]  # "mask_000087" -> "000087"

            # Try possible image name patterns
            candidates = [
                core + mext,             # "000087.png"
                "input_" + core + mext,  # "input_000087.png"
            ]

            img_path = None
            for cname in candidates:
                cand_path = os.path.join(image_dir, cname)
                if os.path.exists(cand_path):
                    img_path = cand_path
                    break

            if img_path is None:
                raise FileNotFoundError(
                    f"Could not find matching image for mask '{mname}' "
                    f"under '{image_dir}'. Tried: {candidates}"
                )

            mask_path = os.path.join(mask_dir, mname)
            self.samples.append((img_path, mask_path))

        print("Flare7KSegDataset: first 3 samples:")
        for i in range(min(3, len(self.samples))):
            print("  ", self.samples[i])


    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> torch.Tensor:
        img = self.Image.open(path).convert("RGB")
        if self.img_size is not None:
            img = img.resize((self.img_size, self.img_size), self.Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0  # [H,W,3]
        return torch.from_numpy(arr).permute(2, 0, 1)  # [3,H,W]

    def _load_mask(self, path: str) -> torch.Tensor:
        img = self.Image.open(path).convert("L")
        if self.img_size is not None:
            img = img.resize((self.img_size, self.img_size), self.Image.NEAREST)
        arr = np.array(img, dtype=np.float32) / 255.0  # [H,W] in [0,1]
        arr = (arr > 0.5).astype(np.float32)           # binarize
        return torch.from_numpy(arr).unsqueeze(0)      # [1,H,W]

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]

        frame = self._load_image(img_path)   # [3,H,W]
        mask  = self._load_mask(mask_path)   # [1,H,W]

        # To keep train.py unchanged, return (clean_frames, frames, masks)
        # Here we don't distinguish "clean" vs "glare" frame, so we just
        # return the same tensor twice.
        return frame, frame.clone(), mask


# ------------------------------------------------------------
# Entry aggregation / manifest handling
# ------------------------------------------------------------

def _load_entries_from_manifests(manifests: List[str]) -> List[dict]:
    if load_manifest is None:
        raise ImportError("load_manifest not available. Implement it in real_data.py or provide manifests another way.")
    entries: List[dict] = []
    for m in manifests:
        mpath = _resolve_path(m)
        items = load_manifest(mpath, require_exists=True, filters=None)
        entries.extend(items)
    if not entries:
        raise RuntimeError("No entries found from provided manifests.")
    return entries


# ------------------------------------------------------------
# Public factory
# ------------------------------------------------------------

# Simple directory scanner for BDD100K-style layout
# root_dir
# └── 100k
#     ├── train
#     ├── val
#     └── test

def _scan_bdd100k_like(root_dir: str, split: str) -> List[dict]:
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be one of train/val/test, got {split}")
    base = os.path.join(root_dir, split)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Split folder not found: {base}")
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    entries: List[dict] = []
    for root, _, files in os.walk(base):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in exts:
                entries.append({"path": os.path.join(root, fn)})
    if not entries:
        raise RuntimeError(f"No images found under {base}")
    entries.sort(key=lambda d: d["path"])  # deterministic order
    return entries


def get_dataset(config_path: str = "config.json"):
    """Selects and constructs the dataset used by train.py.

    New/updated config keys:
      data.source: one of
        - "glare_synth_online"   (self‑supervised, recommended)
        - "real_glare"            (if you have RealGlareDataset)
        - "images_zero_mask"      (fallback sanity)
        - legacy: "bdd100k", "bdd100k_images_only", "synthetic" (still supported where possible)

      data.clean_manifests: ["bdd100k_manifest.json", ...] (optional)
      data.root_dir: "C:/.../bdd100k_images_100k/100k" (optional directory scan)
      data.split: "train" | "val" | "test" (used with root_dir)
      data.img_size: int (optional)
      data.sequence_len: (unused here, kept for compatibility)
      data.synth: dict with GlareSynthOnline params (see class docstring)
      data.glare: (forwarded to RealGlareDataset when used)
    """

    cfg = _load_config(config_path)
    data_cfg = cfg.get("data", {})
    source = str(data_cfg.get("source", "glare_synth_online")).lower()

    img_size = data_cfg.get("img_size", None)
    seq_len = data_cfg.get("sequence_len", 1)  # kept for compatibility

    # If a root_dir is provided, prefer scanning it (supports pre-split folders)
    entries: List[dict] = []
    root_dir = data_cfg.get("root_dir")
    split = str(data_cfg.get("split", "train")).lower()
    if root_dir:
        entries = _scan_bdd100k_like(_resolve_path(root_dir), split)
    else:
        # Resolve manifests (prefer multi‑mix if provided)
        if "clean_manifests" in data_cfg:
            manifests = [_resolve_path(p) for p in data_cfg.get("clean_manifests", [])]
        else:
            # Back‑compat single manifest
            manifests = [_resolve_path(data_cfg.get("manifest_path", "bdd100k_manifest.json"))]
        entries = _load_entries_from_manifests(manifests)

    if source in ("glare_synth_online", "glare_synthesis_online"):
        synth_cfg = data_cfg.get("synth", {})
        ds = GlareSynthOnline(entries=entries, img_size=img_size, synth_cfg=synth_cfg)
        # Flag that this dataset already respects a folder split
        ds.pre_split = split  # type: ignore[attr-defined]
        return ds

    if source == "real_glare":
        if RealGlareDataset is None:
            raise ImportError("RealGlareDataset not available. Set data.source to 'glare_synth_online' or implement RealGlareDataset.")
        return RealGlareDataset(
            manifest_path=root_dir or (data_cfg.get("manifest_path") or ""),
            sequence_len=seq_len,
            img_size=img_size,
            glare_config=data_cfg.get("glare", {}),
        )

    # Legacy routes — maintain backwards compatibility with earlier configs
    if source == "bdd100k":
        if RealGlareDataset is not None:
            return RealGlareDataset(
                manifest_path=root_dir or (data_cfg.get("manifest_path") or ""),
                sequence_len=seq_len,
                img_size=img_size,
                glare_config=data_cfg.get("glare", {}),
            )
        print("Warning: RealGlareDataset not available, falling back to images with zero masks.")
        return _ImagesWithZeroMask(entries=entries, img_size=img_size)

    if source == "bdd100k_images_only":
        if RealImageDataset is None:
            raise ImportError("real_data.RealImageDataset not available.")
        return RealImageDataset(entries, img_size=img_size)

    if source == "images_zero_mask":
        return _ImagesWithZeroMask(entries=entries, img_size=img_size)
    
    if source == "flare7k":
        flare_cfg = data_cfg.get("flare7k", {})
        image_dir = _resolve_path(flare_cfg["image_dir"])
        mask_dir  = _resolve_path(flare_cfg["mask_dir"])
        return Flare7KSegDataset(image_dir=image_dir, mask_dir=mask_dir, img_size=img_size)

    raise ValueError(f"Unknown data source: {source}")
