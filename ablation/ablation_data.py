from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config import DATA, PARAMETERS, TEST, Variant

def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

base = _load("ablation_base_generator", TEST / "data_generator2.0.py")
v3 = _load("ablation_v3_generator", TEST / "data_generator3.py")


def _hard_band_masks(center_x: float, delta: float) -> np.ndarray:
    """Return five disjoint theoretical band masks with no boundary smoothing."""
    distance = np.abs(np.arange(v3.IMG_W, dtype=np.float32) - np.float32(center_x))
    edges = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float32) * delta
    masks = [
        ((distance >= inner) & (distance < outer)).astype(np.float32)
        for inner, outer in zip(edges[:-1], edges[1:])
    ]
    return np.stack(
        [np.broadcast_to(mask, (v3.IMG_H, v3.IMG_W)) for mask in masks]
    ).astype(np.float32)


def build_variant_input(
    image: np.ndarray,
    parameters: np.ndarray,
    variant: Variant,
    rng: np.random.Generator,
    *,
    center_offset_pixels: float = 0.0,
    delta_offset_pixels: float = 0.0,
    geometry_jitter: bool = True,
) -> np.ndarray:
    """Build a fair variant input from inference-available mask geometry only.

    ``center_offset_pixels`` and ``delta_offset_pixels`` are evaluation-only
    perturbations.  They do not alter labels or the underlying OD image.
    """
    if variant.channels == 1:
        return image[None]
    if geometry_jitter:
        center, delta, softness = v3.jittered_mask_geometry(parameters, rng)
    else:
        center = v3.IMG_W / 2.0 + float(parameters[9])
        delta = float(parameters[7])
        softness = v3.DEFAULT_MASK_SOFTNESS_PIXELS
    center += variant.mask_center_offset_pixels + center_offset_pixels
    delta = max(1.0, delta + delta_offset_pixels)
    if variant.mask_mode == "soft":
        return v3.build_six_channel_input(image, center, delta, softness)
    if variant.mask_mode == "hard":
        return np.concatenate([image[None], image[None] * _hard_band_masks(center, delta)]).astype(np.float32)
    raise ValueError(f"Unsupported mask_mode: {variant.mask_mode}")


class AblationDataset(Dataset):
    """One deterministic parameter split with online two-view augmentation.

    Validation has exactly 250 examples each of held-out real, PCA, and mixed
    noise.  It never depends on which model is being evaluated.
    """
    def __init__(self, variant: Variant, split: str, seed: int, epoch: int = 0):
        if split not in {"train", "validation"}:
            raise ValueError(split)
        bank = np.load(PARAMETERS)
        wanted = 0 if split == "train" else 1
        self.indices = np.flatnonzero(bank["split"] == wanted)
        self.parameters = bank["parameters"]
        self.variant, self.split, self.seed, self.epoch = variant, split, seed, epoch
        self.augmentor = base.NoiseAugmentor(DATA, seed=seed)
        if split == "validation" and len(self.indices) != 750:
            raise ValueError("The fixed benchmark requires exactly 750 validation examples.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch if self.split == "train" else 0

    def __len__(self) -> int:
        return len(self.indices)

    def _rng(self, global_index: int, view: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, global_index, view]))

    def _noise(self, clean: np.ndarray, local_index: int, rng: np.random.Generator) -> np.ndarray:
        tier = ("low", "medium", "high")[local_index % 3] if self.split == "validation" else None
        # A0 differs only in its *training* augmentation.  Validation is the
        # common held-out real/PCA/mixed benchmark for every variant.
        if self.variant.noise_policy == "white" and self.split == "train":
            ratio_low, ratio_high = self.augmentor.TIER_RATIOS[tier or ("low", "medium", "high")[int(rng.integers(0, 3))]]
            noise = rng.normal(size=clean.shape).astype(np.float32)
            noise = base._standardize(noise) * max(float(np.std(clean)), .02) * float(rng.uniform(ratio_low, ratio_high))
            return (clean + noise).astype(np.float32)
        # The benchmark explicitly balances raw/PCA/mixed. Raw uses the held-out source.
        if self.split == "validation":
            mode = ("raw", "pca", "mixed")[local_index // 250]
        else:
            mode = str(rng.choice(["raw", "pca", "mixed"], p=[.30, .30, .40]))
        raw = self.augmentor._raw_noise(self.split, rng)
        pca = self.augmentor._pca_noise(rng)
        white = base._standardize(rng.normal(size=clean.shape))
        components = {"raw": raw, "pca": pca, "mixed": .50 * raw + .30 * pca + .20 * white}
        base_noise = base._standardize(components[mode])
        picked = tier or ("low", "medium", "high")[int(rng.integers(0, 3))]
        lo, hi = self.augmentor.TIER_RATIOS[picked]
        return (clean + base_noise * max(float(np.std(clean)), .02) * float(rng.uniform(lo, hi))).astype(np.float32)

    def _input(self, image: np.ndarray, parameters: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return build_variant_input(image, parameters, self.variant, rng)

    def __getitem__(self, local_index: int):
        idx = int(self.indices[local_index]); params = self.parameters[idx]
        clean = base.render_clean_image(params)
        first_rng, second_rng = self._rng(idx, 0), self._rng(idx, 1)
        first = self._input(self._noise(clean, local_index, first_rng), params, first_rng)
        second = self._input(self._noise(clean, local_index, second_rng), params, second_rng)
        return torch.from_numpy(first), torch.from_numpy(second), torch.from_numpy(params[:5].astype(np.float32))
