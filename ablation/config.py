from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
# The ablation package lives beside the V3 source in this repository.
# Laboratory images are NOT distributed; point ORBITAL_DATA_ROOT at your own
# data directory (containing BandMapping_data/ and Extracted_Noise_Bank/).
TEST = PROJECT
DATA = Path(
    os.environ.get(
        "ORBITAL_DATA_ROOT",
        str(PROJECT / "data"),
    )
)
PARAMETERS = Path(
    os.environ.get(
        "ORBITAL_PARAMETERS",
        str(DATA / "training_v2" / "theory_parameters_5000.npz"),
    )
)
RESULTS = ROOT / "results"
SEEDS = (20260727, 20260728, 20260729)
BANDS = ("s", "p", "d", "f", "g")

@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    channels: int
    noise_policy: str
    consistency_weight: float
    mask_mode: str = "none"
    mask_center_offset_pixels: float = 0.0

VARIANTS = (
    Variant("A0", "A0: single-channel baseline (white noise)", 1, "white", 0.0),
    Variant("A1", "A1: + real/PCA noise", 1, "real_pca", 0.0),
    Variant("A2", "A2: + consistency loss", 1, "real_pca", 0.2),
    Variant("A3", "A3: soft masks + mask jitter", 6, "real_pca", 0.2, "soft"),
    Variant("A4", "A4: hard masks + mask jitter", 6, "real_pca", 0.2, "hard"),
    Variant("A5", "A5: soft masks, deliberately +12 px shifted", 6, "real_pca", 0.2, "soft", 12.0),
)
V3_VARIANT = next(variant for variant in VARIANTS if variant.key == "A3")
