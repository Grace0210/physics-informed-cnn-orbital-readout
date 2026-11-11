#!/usr/bin/env python3
"""V3 online sim-to-real generator with physics-guided soft band channels.

This version stores only 5,000 clean parameter combinations. During training,
each clean theory image receives two newly sampled noise realizations per epoch.
Training noise can contain held-in raw experimental residual patches, PCA
reconstructions, and white noise. Validation uses held-out raw residual sources.

Each noisy image is converted to six channels: the full image followed by the
same noisy image multiplied by soft s/p/d/f/g spatial masks. Mask geometry is
jittered online so the CNN cannot rely on perfectly known boundaries.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.special import erf
from torch.utils.data import Dataset


IMG_H = 400
IMG_W = 300
ROI = (310, 710, 555, 855)
DEFAULT_SEED = 20260727
DELTA_MIN_PIXELS = 23.0
DELTA_MAX_PIXELS = 25.0
DEFAULT_MASK_DELTA_PIXELS = 26.0
DEFAULT_MASK_SOFTNESS_PIXELS = 4.0
CHANNEL_NAMES = ["full", "s", "p", "d", "f", "g"]
PARAMETER_NAMES = [
    "s",
    "p",
    "d",
    "f",
    "g",
    "t_tf",
    "abs_t_scale",
    "Delta_pixels",
    "sigma_pixels",
    "xc_pixels",
    "yc_pixels",
    "peak_od",
    "thermal_ratio",
    "thermal_sigma_pixels",
]


def _standardize(array: NDArray[np.floating]) -> NDArray[np.float32]:
    result = np.asarray(array, dtype=np.float32)
    result = result - np.mean(result)
    scale = float(np.std(result))
    if scale < 1e-8:
        raise ValueError("Cannot standardize a constant noise image.")
    return (result / scale).astype(np.float32)


def _band_populations(rng: np.random.Generator) -> NDArray[np.float32]:
    choice = rng.random()
    if choice < 0.2:
        alpha = [6.0, 1.0, 1.0, 0.5, 0.1]
    elif choice < 0.4:
        alpha = [1.0, 6.0, 1.0, 0.5, 0.1]
    elif choice < 0.6:
        alpha = [1.0, 2.0, 6.0, 2.0, 1.0]
    elif choice < 0.8:
        alpha = [1.0, 1.0, 2.0, 5.0, 5.0]
    else:
        alpha = [1.0, 1.0, 1.0, 1.0, 1.0]
    return rng.dirichlet(alpha).astype(np.float32)


def create_parameter_bank(
    output_path: Path,
    sample_count: int = 5000,
    validation_fraction: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> None:
    """Create disjoint train/validation theory parameter combinations."""
    rng = np.random.default_rng(seed)
    parameters = np.zeros(
        (sample_count, len(PARAMETER_NAMES)),
        dtype=np.float32,
    )
    for index in range(sample_count):
        populations = _band_populations(rng)
        parameters[index] = [
            *populations,
            rng.uniform(0.1, 2.0),
            rng.uniform(1.5, 4.0),
            rng.uniform(DELTA_MIN_PIXELS, DELTA_MAX_PIXELS),
            rng.uniform(1.0, 5.0),
            rng.uniform(-10.0, 10.0),
            rng.uniform(-15.0, 15.0),
            rng.uniform(0.2, 1.2),
            rng.uniform(0.0, 0.3),
            rng.uniform(100.0, 250.0),
        ]

    order = rng.permutation(sample_count)
    validation_count = int(round(sample_count * validation_fraction))
    split = np.zeros(sample_count, dtype=np.uint8)
    split[order[-validation_count:]] = 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        parameters=parameters,
        split=split,
        parameter_names=np.asarray(PARAMETER_NAMES),
        seed=np.asarray(seed),
    )


def render_clean_image(parameters: NDArray[np.floating]) -> NDArray[np.float32]:
    """Render the five-band erf theory model without any noise."""
    (
        s,
        p,
        d,
        f,
        g,
        t_tf,
        abs_t_scale,
        delta,
        sigma,
        x_shift,
        y_shift,
        peak_od,
        thermal_ratio,
        thermal_sigma,
    ) = [float(value) for value in parameters]

    x_pixels = np.arange(IMG_W, dtype=np.float32) - IMG_W / 2.0
    y_relative = (
        np.linspace(-3.0, 3.0, IMG_H, dtype=np.float32) - y_shift / 40.0
    )
    y_scaled = y_relative / np.sqrt(abs_t_scale)
    profile_y = t_tf * np.logaddexp(
        0.0,
        (1.5 - y_scaled**2) / (t_tf + 1e-5),
    )
    profile_y /= np.max(profile_y)
    factor = 2.0 * np.sqrt(2.0) * sigma

    def erf_term(amplitude: float, multiple: int) -> NDArray[np.float32]:
        return amplitude / 2.0 * (
            erf((multiple * delta + 2.0 * (x_pixels - x_shift)) / factor)
            + erf(
                (multiple * delta - 2.0 * (x_pixels - x_shift)) / factor
            )
        )

    profile_x = (
        erf_term(s, 1)
        + erf_term(p, -1)
        + erf_term(p, 2)
        + erf_term(d, -2)
        + erf_term(d, 3)
        + erf_term(f, -3)
        + erf_term(f, 4)
        + erf_term(g, -4)
        + erf_term(g, 5)
    )
    profile_x = np.clip(profile_x, 0.0, None)
    core = peak_od * np.outer(profile_y, profile_x)

    x_grid, y_grid = np.meshgrid(
        np.arange(IMG_W, dtype=np.float32),
        np.arange(IMG_H, dtype=np.float32),
    )
    thermal = thermal_ratio * np.max(core) * np.exp(
        -(
            (x_grid - (IMG_W / 2.0 + x_shift)) ** 2
            + (y_grid - (IMG_H / 2.0 + y_shift)) ** 2
        )
        / (2.0 * thermal_sigma**2)
    )
    return (core + thermal).astype(np.float32)


def create_soft_band_masks(
    center_x: float,
    delta: float,
    softness: float = DEFAULT_MASK_SOFTNESS_PIXELS,
    image_height: int = IMG_H,
    image_width: int = IMG_W,
) -> NDArray[np.float32]:
    """Create overlapping soft masks for the five theoretical band regions.

    In the five-band theory model the nominal radial intervals are:
    s=[0, 0.5Δ], p=[0.5Δ, Δ], d=[Δ, 1.5Δ],
    f=[1.5Δ, 2Δ], and g=[2Δ, 2.5Δ].
    Logistic boundaries preserve the blurred signal shared by adjacent bands.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    if softness <= 0:
        raise ValueError("softness must be positive")

    x = np.arange(image_width, dtype=np.float32)
    distance = np.abs(x - np.float32(center_x))

    def sigmoid(values: NDArray[np.float32]) -> NDArray[np.float32]:
        clipped = np.clip(values, -30.0, 30.0)
        return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)

    masks_1d: list[NDArray[np.float32]] = []
    intervals = [
        (0.0, 0.5 * delta),
        (0.5 * delta, 1.0 * delta),
        (1.0 * delta, 1.5 * delta),
        (1.5 * delta, 2.0 * delta),
        (2.0 * delta, 2.5 * delta),
    ]
    for inner, outer in intervals:
        outer_gate = sigmoid((np.float32(outer) - distance) / softness)
        if inner == 0.0:
            mask_1d = outer_gate
        else:
            inner_gate = sigmoid((distance - np.float32(inner)) / softness)
            mask_1d = inner_gate * outer_gate
        masks_1d.append(mask_1d.astype(np.float32))

    return np.stack(
        [np.broadcast_to(mask, (image_height, image_width)) for mask in masks_1d]
    ).astype(np.float32)


def build_six_channel_input(
    image: NDArray[np.floating],
    center_x: float | None = None,
    delta: float = DEFAULT_MASK_DELTA_PIXELS,
    softness: float = DEFAULT_MASK_SOFTNESS_PIXELS,
) -> NDArray[np.float32]:
    """Return [full, s, p, d, f, g] from one already-noised image."""
    image_float = np.asarray(image, dtype=np.float32)
    if image_float.shape != (IMG_H, IMG_W):
        raise ValueError(
            f"Expected image shape {(IMG_H, IMG_W)}, got {image_float.shape}"
        )
    if center_x is None:
        center_x = IMG_W / 2.0
    masks = create_soft_band_masks(center_x, delta, softness)
    return np.concatenate(
        [image_float[None], image_float[None] * masks],
        axis=0,
    ).astype(np.float32)


def jittered_mask_geometry(
    parameters: NDArray[np.floating],
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Sample realistic mask uncertainty independently for each noisy view."""
    true_delta = float(parameters[7])
    true_center_x = IMG_W / 2.0 + float(parameters[9])
    center_x = true_center_x + float(rng.uniform(-4.0, 4.0))
    delta = float(
        np.clip(
            true_delta + rng.uniform(-2.0, 4.0),
            21.0,
            30.0,
        )
    )
    softness = float(rng.uniform(3.0, 6.0))
    return center_x, delta, softness


@dataclass
class RawNoiseSource:
    name: str
    residual: NDArray[np.float32]


def discover_raw_noise_sources(data_root: Path) -> list[RawNoiseSource]:
    """Extract high-frequency residuals from real atom-free Reference frames."""
    sources: list[RawNoiseSource] = []
    y1, y2, x1, x2 = ROI
    for background_path in sorted(
        data_root.glob("BandMapping_data/**/BG/BG_r.tif")
    ):
        scan_dir = background_path.parent.parent
        background = np.asarray(
            Image.open(background_path),
            dtype=np.float32,
        )
        for reference_path in sorted(scan_dir.glob("*AveR.tif")):
            reference = np.asarray(
                Image.open(reference_path),
                dtype=np.float32,
            )
            if reference.shape != background.shape:
                continue
            pure_reference = reference - background
            patch = pure_reference[y1:y2, x1:x2]
            if patch.shape != (IMG_H, IMG_W):
                continue
            # Remove slowly varying probe illumination; retain camera/fringe
            # residual texture as an empirical noise patch.
            residual = patch - gaussian_filter(patch, sigma=20)
            sources.append(
                RawNoiseSource(
                    name=str(reference_path.relative_to(data_root)),
                    residual=_standardize(residual),
                )
            )
    if len(sources) < 2:
        raise RuntimeError(
            "Need at least two real Reference noise sources to make a "
            "source-disjoint train/validation split."
        )
    return sources


class NoiseAugmentor:
    TIER_RATIOS = {
        "low": (0.10, 0.20),
        "medium": (0.25, 0.45),
        "high": (0.50, 0.80),
    }

    def __init__(
        self,
        data_root: Path,
        seed: int = DEFAULT_SEED,
        validation_fraction: float = 0.20,
    ) -> None:
        self.seed = seed
        raw_sources = discover_raw_noise_sources(data_root)
        split_rng = np.random.default_rng(seed)
        order = split_rng.permutation(len(raw_sources))
        validation_count = max(
            1,
            int(round(len(raw_sources) * validation_fraction)),
        )
        validation_indices = set(order[-validation_count:].tolist())
        self.train_raw = [
            source
            for index, source in enumerate(raw_sources)
            if index not in validation_indices
        ]
        self.validation_raw = [
            source
            for index, source in enumerate(raw_sources)
            if index in validation_indices
        ]

        pca_path = (
            data_root
            / "Extracted_Noise_Bank"
            / "global_pca_fringes.npy"
        )
        components = np.load(pca_path)
        if components.shape != (5, IMG_H, IMG_W):
            raise ValueError(f"Unexpected PCA shape: {components.shape}")
        self.pca_components = np.stack(
            [_standardize(component) for component in components]
        )

    @staticmethod
    def _transform_raw(
        residual: NDArray[np.float32],
        rng: np.random.Generator,
    ) -> NDArray[np.float32]:
        transformed = residual
        if rng.random() < 0.5:
            transformed = np.flip(transformed, axis=0)
        if rng.random() < 0.5:
            transformed = np.flip(transformed, axis=1)
        transformed = np.roll(
            transformed,
            shift=int(rng.integers(-30, 31)),
            axis=0,
        )
        transformed = np.roll(
            transformed,
            shift=int(rng.integers(-30, 31)),
            axis=1,
        )
        return _standardize(transformed)

    def _raw_noise(
        self,
        split: str,
        rng: np.random.Generator,
    ) -> NDArray[np.float32]:
        sources = (
            self.train_raw if split == "train" else self.validation_raw
        )
        source = sources[int(rng.integers(0, len(sources)))]
        return self._transform_raw(source.residual, rng)

    def _pca_noise(
        self,
        rng: np.random.Generator,
    ) -> NDArray[np.float32]:
        coefficients = rng.normal(0.0, 1.0, len(self.pca_components))
        mixture = np.tensordot(
            coefficients,
            self.pca_components,
            axes=1,
        )
        return _standardize(mixture)

    @staticmethod
    def tier_probabilities(epoch: int) -> NDArray[np.float64]:
        # Noise curriculum: begin mostly clean, then expose all SNR levels.
        if epoch < 10:
            return np.asarray([0.60, 0.30, 0.10])
        if epoch < 30:
            return np.asarray([0.30, 0.50, 0.20])
        return np.asarray([1.0 / 3.0] * 3)

    def apply(
        self,
        clean: NDArray[np.float32],
        split: str,
        rng: np.random.Generator,
        epoch: int,
        forced_tier: str | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, float | str]]:
        tiers = ["low", "medium", "high"]
        tier = (
            forced_tier
            if forced_tier is not None
            else str(rng.choice(tiers, p=self.tier_probabilities(epoch)))
        )

        if split == "train":
            mode = str(
                rng.choice(
                    ["raw", "pca", "mixed"],
                    p=[0.30, 0.30, 0.40],
                )
            )
        else:
            # Validation deliberately uses held-out raw files and never the
            # training raw sources. Keep it fixed/reproducible at epoch zero.
            mode = str(rng.choice(["raw", "raw_white"], p=[0.60, 0.40]))

        raw = self._raw_noise(split, rng)
        pca = self._pca_noise(rng)
        white = _standardize(rng.normal(size=(IMG_H, IMG_W)))
        if mode == "raw":
            base_noise = raw
        elif mode == "pca":
            base_noise = pca
        elif mode == "raw_white":
            base_noise = 0.75 * raw + 0.25 * white
        else:
            base_noise = 0.50 * raw + 0.30 * pca + 0.20 * white
        base_noise = _standardize(base_noise)

        clean_rms = max(float(np.std(clean)), 0.02)
        ratio_low, ratio_high = self.TIER_RATIOS[tier]
        noise_to_signal = float(rng.uniform(ratio_low, ratio_high))
        noise_rms = clean_rms * noise_to_signal
        noise = base_noise * noise_rms
        noisy = (clean + noise).astype(np.float32)
        return noisy, {
            "tier": tier,
            "mode": mode,
            "noise_rms": float(np.std(noise)),
            "noise_to_signal": noise_to_signal,
        }

    def manifest(self) -> dict[str, object]:
        return {
            "train_raw_sources": [source.name for source in self.train_raw],
            "validation_raw_sources": [
                source.name for source in self.validation_raw
            ],
            "pca_components": int(len(self.pca_components)),
            "tier_noise_to_signal_ranges": self.TIER_RATIOS,
            "train_noise_modes": ["raw", "pca", "mixed"],
            "validation_noise_modes": ["raw", "raw_white"],
        }


class OnlineTwoViewDataset(Dataset):
    """Return two independently noised six-channel views of one theory image."""

    def __init__(
        self,
        parameter_bank_path: Path,
        data_root: Path,
        split: str,
        seed: int = DEFAULT_SEED,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        bank = np.load(parameter_bank_path)
        parameters = bank["parameters"]
        split_flags = bank["split"]
        desired_flag = 0 if split == "train" else 1
        self.global_indices = np.flatnonzero(split_flags == desired_flag)
        self.parameters = parameters
        self.split = split
        self.seed = seed
        self.epoch = 0
        self.augmentor = NoiseAugmentor(data_root, seed=seed)

    def set_epoch(self, epoch: int) -> None:
        # Validation remains fixed for comparable metrics across epochs.
        self.epoch = epoch if self.split == "train" else 0

    def __len__(self) -> int:
        return len(self.global_indices)

    def _rng(self, global_index: int, view: int) -> np.random.Generator:
        seed_sequence = np.random.SeedSequence(
            [self.seed, self.epoch, global_index, view]
        )
        return np.random.default_rng(seed_sequence)

    def __getitem__(
        self,
        local_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_index = int(self.global_indices[local_index])
        parameters = self.parameters[global_index]
        clean = render_clean_image(parameters)
        first, _ = self.augmentor.apply(
            clean,
            self.split,
            self._rng(global_index, 0),
            self.epoch,
        )
        second, _ = self.augmentor.apply(
            clean,
            self.split,
            self._rng(global_index, 1),
            self.epoch,
        )
        first_geometry = jittered_mask_geometry(
            parameters,
            self._rng(global_index, 10),
        )
        second_geometry = jittered_mask_geometry(
            parameters,
            self._rng(global_index, 11),
        )
        first_channels = build_six_channel_input(first, *first_geometry)
        second_channels = build_six_channel_input(second, *second_geometry)
        label = parameters[:5].astype(np.float32)
        return (
            torch.from_numpy(first_channels),
            torch.from_numpy(second_channels),
            torch.from_numpy(label),
        )


def save_channel_preview(
    parameter_bank_path: Path,
    data_root: Path,
    output_path: Path,
    seed: int,
) -> None:
    """Visual check that all six channels share one noisy base image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bank = np.load(parameter_bank_path)
    train_index = int(np.flatnonzero(bank["split"] == 0)[0])
    parameters = bank["parameters"][train_index]
    clean = render_clean_image(parameters)
    augmentor = NoiseAugmentor(data_root, seed=seed)
    noisy, metadata = augmentor.apply(
        clean,
        split="train",
        rng=np.random.default_rng(seed + 300),
        epoch=50,
        forced_tier="medium",
    )
    geometry = jittered_mask_geometry(
        parameters,
        np.random.default_rng(seed + 301),
    )
    channels = build_six_channel_input(noisy, *geometry)

    figure, axes = plt.subplots(2, 3, figsize=(12, 10), constrained_layout=True)
    for axis, channel, name in zip(axes.flat, channels, CHANNEL_NAMES):
        artist = axis.imshow(channel, cmap="jet", origin="lower")
        axis.set_title(name)
        axis.axis("off")
        figure.colorbar(artist, ax=axis, fraction=0.046)
    figure.suptitle(
        "V3 input channels from one shared noisy image\n"
        f"center={geometry[0]:.1f}, delta={geometry[1]:.1f}, "
        f"softness={geometry[2]:.1f}, noise={metadata['mode']}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_preview(
    parameter_bank_path: Path,
    data_root: Path,
    output_path: Path,
    seed: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bank = np.load(parameter_bank_path)
    train_index = int(np.flatnonzero(bank["split"] == 0)[0])
    clean = render_clean_image(bank["parameters"][train_index])
    augmentor = NoiseAugmentor(data_root, seed=seed)
    tiers = ["low", "medium", "high"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    images = [clean]
    noise_maps = [np.zeros_like(clean)]
    titles = ["Clean theory"]
    for tier_index, tier in enumerate(tiers):
        noisy, metadata = augmentor.apply(
            clean,
            split="train",
            rng=np.random.default_rng(seed + 100 + tier_index),
            epoch=50,
            forced_tier=tier,
        )
        images.append(noisy)
        noise_maps.append(noisy - clean)
        titles.append(
            f"{tier.title()} noise\n"
            f"{metadata['mode']}, ratio={metadata['noise_to_signal']:.2f}"
        )

    common_min = min(float(image.min()) for image in images)
    common_max = max(float(image.max()) for image in images)
    for index, (image, title) in enumerate(zip(images, titles)):
        artist = axes[0, index].imshow(
            image,
            cmap="jet",
            origin="lower",
            vmin=common_min,
            vmax=common_max,
        )
        axes[0, index].set_title(title)
        axes[0, index].axis("off")
        fig.colorbar(artist, ax=axes[0, index], fraction=0.046)

        noise = noise_maps[index]
        limit = max(float(np.max(np.abs(noise))), 1e-6)
        artist = axes[1, index].imshow(
            noise,
            cmap="seismic",
            origin="lower",
            vmin=-limit,
            vmax=limit,
        )
        axes[1, index].set_title(f"Added noise, RMS={np.std(noise):.4f}")
        axes[1, index].axis("off")
        fig.colorbar(artist, ax=axes[1, index], fraction=0.046)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_dir / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "data" / "training_v3",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameter_path = args.output_dir / "theory_parameters_5000.npz"
    create_parameter_bank(
        parameter_path,
        sample_count=args.samples,
        seed=args.seed,
    )
    augmentor = NoiseAugmentor(args.data_root, seed=args.seed)
    (args.output_dir / "noise_split_manifest.json").write_text(
        json.dumps(augmentor.manifest(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_preview(
        parameter_path,
        args.data_root,
        args.output_dir / "online_noise_preview.png",
        args.seed,
    )
    save_channel_preview(
        parameter_path,
        args.data_root,
        args.output_dir / "six_channel_preview.png",
        args.seed,
    )
    print(f"Saved parameter bank: {parameter_path}")
    print(
        f"Train/validation counts: "
        f"{int(np.sum(np.load(parameter_path)['split'] == 0))}/"
        f"{int(np.sum(np.load(parameter_path)['split'] == 1))}"
    )
    print(
        "No fixed image dataset was written. Noise and six-channel soft masks "
        "are generated online in OnlineTwoViewDataset."
    )


if __name__ == "__main__":
    main()
