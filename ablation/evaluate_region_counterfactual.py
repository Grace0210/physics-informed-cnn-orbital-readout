#!/usr/bin/env python3
"""Test whether each output depends mainly on its physical band region.

For each synthetic validation image, one disjoint theoretical band interval is
replaced by the image mean (zero after standardisation).  The script records
the resulting five-by-five output-change and absolute-error-change matrices.
It is a counterfactual sensitivity test, not a real-data accuracy claim.
"""
from __future__ import annotations

import csv
import json

import numpy as np
import torch

from ablation_data import AblationDataset, _hard_band_masks, base, build_variant_input
from config import BANDS, RESULTS, SEEDS, VARIANTS
from run_ablation import device, model_for


def predict(model, encoded: np.ndarray, dev: torch.device) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(encoded[None]).to(dev)).cpu().numpy()[0]


def main() -> None:
    dev = device()
    output = RESULTS / "region_counterfactual"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    matrices = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            checkpoint = RESULTS / variant.key / f"seed_{seed}" / "best_model.pth"
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing {checkpoint}; run the ablation first.")
            model = model_for(variant).to(dev)
            model.load_state_dict(torch.load(checkpoint, map_location=dev))
            model.eval()
            dataset = AblationDataset(variant, "validation", seed)
            output_delta = np.zeros((5, 5), dtype=np.float64)
            error_delta = np.zeros((5, 5), dtype=np.float64)
            for local_index, global_index in enumerate(dataset.indices):
                params = dataset.parameters[int(global_index)]
                noise_rng = dataset._rng(int(global_index), 0)
                image = dataset._noise(base.render_clean_image(params), local_index, noise_rng)
                clean_input = build_variant_input(
                    image, params, variant, np.random.default_rng([seed, int(global_index), 1]), geometry_jitter=False
                )
                clean_prediction = predict(model, clean_input, dev)
                masks = _hard_band_masks(image.shape[1] / 2.0 + float(params[9]), float(params[7]))
                for region_index, mask in enumerate(masks):
                    perturbed = image * (1.0 - mask)
                    encoded = build_variant_input(
                        perturbed, params, variant, np.random.default_rng([seed, int(global_index), 2]), geometry_jitter=False
                    )
                    prediction = predict(model, encoded, dev)
                    output_delta[region_index] += np.abs(prediction - clean_prediction)
                    error_delta[region_index] += np.abs(prediction - params[:5]) - np.abs(clean_prediction - params[:5])
            output_delta /= len(dataset)
            error_delta /= len(dataset)
            matrices[f"{variant.key}_seed_{seed}"] = {
                "mean_absolute_prediction_change": output_delta.tolist(),
                "mean_absolute_error_change": error_delta.tolist(),
            }
            for region_index, region in enumerate(BANDS):
                off_target = np.delete(output_delta[region_index], region_index)
                rows.append({
                    "variant": variant.key,
                    "seed": seed,
                    "occluded_region": region,
                    "target_output_change": float(output_delta[region_index, region_index]),
                    "off_target_mean_change": float(off_target.mean()),
                    "selectivity_ratio": float(output_delta[region_index, region_index] / max(off_target.mean(), 1e-12)),
                    "target_mae_change": float(error_delta[region_index, region_index]),
                })
    with (output / "per_seed_selectivity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for variant in VARIANTS:
        part = [row for row in rows if row["variant"] == variant.key]
        summary.append({
            "variant": variant.key,
            "mean_selectivity_ratio": float(np.mean([row["selectivity_ratio"] for row in part])),
            "mean_target_mae_change": float(np.mean([row["target_mae_change"] for row in part])),
        })
    (output / "summary.json").write_text(json.dumps({
        "scope": "Synthetic counterfactual test. Rows: occluded physical region; columns: output band.",
        "summary": summary,
        "matrices": matrices,
    }, indent=2))


if __name__ == "__main__":
    main()
