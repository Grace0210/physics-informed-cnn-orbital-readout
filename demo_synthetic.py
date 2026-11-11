#!/usr/bin/env python3
"""Run the V3 pipeline end to end on purely synthetic data.

No laboratory images are needed. The script samples theory parameters, renders a
clean absorption-density image, adds white noise, builds the six-channel
[full, s, p, d, f, g] soft-mask input, and runs one forward pass of the V3 CNN.
It optionally saves a figure of the six input channels.

    python demo_synthetic.py --figure docs/six_channel_synthetic.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data_generator3 import (
    CHANNEL_NAMES,
    IMG_W,
    _band_populations,
    build_six_channel_input,
    render_clean_image,
)
from model_v3 import AdvancedLatticeCNN


def sample_parameters(rng: np.random.Generator) -> np.ndarray:
    populations = _band_populations(rng)
    return np.asarray(
        [
            *populations,
            rng.uniform(0.1, 2.0),   # t_tf
            rng.uniform(1.5, 4.0),   # abs_t_scale
            rng.uniform(23.0, 25.0), # Delta (px)
            rng.uniform(1.0, 5.0),   # sigma (px)
            rng.uniform(-10.0, 10.0),
            rng.uniform(-15.0, 15.0),
            rng.uniform(0.2, 1.2),   # peak OD
            rng.uniform(0.0, 0.3),   # thermal ratio
            rng.uniform(100.0, 250.0),
        ],
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    params = sample_parameters(rng)
    clean = render_clean_image(params)
    noisy = clean + args.noise * float(clean.max()) * rng.standard_normal(clean.shape).astype(np.float32)
    six = build_six_channel_input(noisy, center_x=IMG_W / 2.0 + float(params[9]), delta=float(params[7]))

    model = AdvancedLatticeCNN().eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(six)[None]).numpy()[0]

    print("input shape      :", six.shape)
    print("true  [s,p,d,f,g]:", np.round(params[:5], 3))
    print("model [s,p,d,f,g]:", np.round(pred, 3), "(untrained weights; sum =", round(float(pred.sum()), 4), ")")
    print("parameters       :", sum(p.numel() for p in model.parameters()))

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 6, figsize=(15, 3.4))
        for ax, channel, name in zip(axes, six, CHANNEL_NAMES):
            ax.imshow(channel, cmap="magma", aspect="auto")
            ax.set_title(name)
            ax.axis("off")
        fig.suptitle("Synthetic absorption-density image and its five soft band-mask channels")
        fig.tight_layout()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=150)
        print("saved", args.figure)


if __name__ == "__main__":
    main()
