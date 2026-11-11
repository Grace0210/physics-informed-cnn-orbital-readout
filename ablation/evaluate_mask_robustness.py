#!/usr/bin/env python3
"""Evaluate mask-geometry robustness after ``run_ablation.py --all``.

This is intentionally synthetic-only: it perturbs inference-time centre and
band-spacing estimates while keeping each rendered image and its five-fraction
ground truth fixed.  It therefore tests whether a claimed gain depends on
perfect mask geometry without pretending to establish real-data accuracy.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
import torch

from ablation_data import AblationDataset, base, build_variant_input
from config import RESULTS, SEEDS, VARIANTS
from run_ablation import device, metrics, model_for


CONDITIONS = (
    ("nominal", 0.0, 0.0),
    ("center_minus_8px", -8.0, 0.0),
    ("center_plus_8px", 8.0, 0.0),
    ("delta_minus_4px", 0.0, -4.0),
    ("delta_plus_4px", 0.0, 4.0),
)


def evaluate_variant(variant, seed: int) -> list[dict[str, object]]:
    checkpoint = RESULTS / variant.key / f"seed_{seed}" / "best_model.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing {checkpoint}; run the controlled ablation first.")
    dataset = AblationDataset(variant, "validation", seed)
    model = model_for(variant).to(device())
    model.load_state_dict(torch.load(checkpoint, map_location=device()))
    model.eval()
    rows: list[dict[str, object]] = []
    for condition, center_offset, delta_offset in CONDITIONS:
        truth, prediction = [], []
        nonfinite = 0
        with torch.no_grad():
            for local_index, global_index in enumerate(dataset.indices):
                params = dataset.parameters[int(global_index)]
                # Use one deterministic noisy view for every geometry condition.
                noise_rng = dataset._rng(int(global_index), 0)
                clean = base.render_clean_image(params)
                image = dataset._noise(clean, local_index, noise_rng)
                input_rng = np.random.default_rng(
                    np.random.SeedSequence([seed, int(global_index), 991])
                )
                encoded = build_variant_input(
                    image,
                    params,
                    variant,
                    input_rng,
                    center_offset_pixels=center_offset,
                    delta_offset_pixels=delta_offset,
                    geometry_jitter=False,
                )
                output = model(torch.from_numpy(encoded[None]).to(device())).cpu().numpy()[0]
                if not np.isfinite(output).all():
                    nonfinite += 1
                    continue
                truth.append(params[:5])
                prediction.append(output)
        row: dict[str, object] = {
            "variant": variant.key,
            "seed": seed,
            "condition": condition,
            "center_offset_pixels": center_offset,
            "delta_offset_pixels": delta_offset,
            "n_total": len(dataset),
            "n_nonfinite": nonfinite,
            "failure_rate": nonfinite / len(dataset),
            **metrics(np.asarray(truth), np.asarray(prediction)),
        }
        rows.append(row)
    return rows


def main() -> None:
    rows = [row for variant in VARIANTS for seed in SEEDS for row in evaluate_variant(variant, seed)]
    output = RESULTS / "mask_robustness"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_seed_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["variant"]), str(row["condition"]))].append(row)
    summary = []
    for (variant, condition), group in grouped.items():
        summary.append(
            {
                "variant": variant,
                "condition": condition,
                "macro_mae_mean": float(np.mean([float(x["macro_mae"]) for x in group])),
                "macro_mae_std": float(np.std([float(x["macro_mae"]) for x in group], ddof=1)),
                "macro_rmse_mean": float(np.mean([float(x["macro_rmse"]) for x in group])),
                "failure_rate_mean": float(np.mean([float(x["failure_rate"]) for x in group])),
            }
        )
    summary.sort(key=lambda item: (item["variant"], CONDITIONS.index(next(c for c in CONDITIONS if c[0] == item["condition"]))))
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "scope": "Synthetic geometry-robustness evaluation only; no real-image labels were used.",
                "conditions": [
                    {"name": name, "center_offset_pixels": center, "delta_offset_pixels": delta}
                    for name, center, delta in CONDITIONS
                ],
                "summary": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
