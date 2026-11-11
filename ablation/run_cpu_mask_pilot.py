#!/usr/bin/env python3
"""A bounded CPU pilot for the soft-mask hypothesis.

This is deliberately not a publication-grade replacement for the full study:
it uses one seed, a stratified 300-image validation subset, 768 training
examples, and 4x average-pooled inputs.  Its only job is to decide whether A3
is promising enough to justify the full-resolution multi-seed experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset, Subset

from ablation_data import AblationDataset
from config import BANDS, RESULTS, VARIANTS
from run_ablation import loss, metrics, model_for


SEED = 20260727
VARIANT_KEYS = ("A2", "A3", "A4", "A5")


class DownsampledViews(Dataset):
    def __init__(self, dataset: Dataset, factor: int = 2) -> None:
        self.dataset = dataset
        self.factor = factor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        first, second, label = self.dataset[index]
        first = F.avg_pool2d(first, self.factor)
        second = F.avg_pool2d(second, self.factor)
        return first, second, label


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stratified_indices(total: int, wanted: int, groups: int) -> list[int]:
    if wanted % groups:
        raise ValueError("wanted must divide evenly across groups")
    per_group = wanted // groups
    group_size = total // groups
    return [
        group * group_size + int(position)
        for group in range(groups)
        for position in np.linspace(0, group_size - 1, per_group, dtype=int)
    ]


def evaluate(model, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    truth, prediction = [], []
    model.eval()
    with torch.no_grad():
        for first, second, label in loader:
            pred = 0.5 * (model(first) + model(second))
            truth.append(label.numpy())
            prediction.append(pred.numpy())
    return np.concatenate(truth), np.concatenate(prediction)


def train_variant(variant, epochs: int, batch_size: int, train_size: int, valid_size: int, factor: int) -> dict[str, object]:
    set_seed(SEED)
    raw_train = AblationDataset(variant, "train", SEED)
    raw_valid = AblationDataset(variant, "validation", SEED)
    train_indices = np.linspace(0, len(raw_train) - 1, train_size, dtype=int).tolist()
    valid_indices = stratified_indices(len(raw_valid), valid_size, groups=3)
    train = DownsampledViews(Subset(raw_train, train_indices), factor)
    valid = DownsampledViews(Subset(raw_valid, valid_indices), factor)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False, num_workers=0)
    model = model_for(variant).cpu()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history, best, best_state, waiting = [], float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        raw_train.set_epoch(epoch)
        for first, second, label in train_loader:
            optimizer.zero_grad()
            total, _, _ = loss(model(first), model(second), label, variant.consistency_weight)
            total.backward()
            optimizer.step()
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for first, second, label in valid_loader:
                total, _, _ = loss(model(first), model(second), label, variant.consistency_weight)
                valid_loss += float(total) * len(label)
        valid_loss /= len(valid)
        history.append({"epoch": epoch + 1, "validation_loss": valid_loss})
        if valid_loss < best:
            best, waiting = valid_loss, 0
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        else:
            waiting += 1
        scheduler.step()
        print(f"{variant.key} epoch={epoch + 1}/{epochs} val={valid_loss:.6f}", flush=True)
        if waiting >= 8:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    truth, prediction = evaluate(model, valid_loader)
    return {
        "variant": variant.key,
        "label": variant.label,
        "device": "cpu",
        "input_resolution": f"{400 // factor}x{300 // factor}",
        "seed": SEED,
        "train_examples": train_size,
        "validation_examples": valid_size,
        "epochs_completed": len(history),
        "best_validation_loss": best,
        **metrics(truth, prediction),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=768)
    parser.add_argument("--valid-size", type=int, default=300)
    parser.add_argument("--downsample-factor", type=int, default=4)
    args = parser.parse_args()
    if args.train_size > 4250 or args.valid_size > 750 or args.valid_size % 3 or args.downsample_factor not in {2, 4}:
        parser.error("train-size must be <=4250; valid-size must be <=750 and divisible by 3; downsample-factor must be 2 or 4")
    selected = [variant for variant in VARIANTS if variant.key in VARIANT_KEYS]
    rows = [train_variant(variant, args.epochs, args.batch_size, args.train_size, args.valid_size, args.downsample_factor) for variant in selected]
    output = RESULTS / "cpu_mask_pilot"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in rows[0] if key != "history"])
        writer.writeheader()
        writer.writerows([{key: value for key, value in row.items() if key != "history"} for row in rows])
    (output / "results.json").write_text(json.dumps({
        "warning": "CPU pilot only: one seed, reduced resolution and subset. Do not report as final evidence.",
        "variants": rows,
        "interpretation": {
            "A3_vs_A2": "Does the correct soft spatial encoding help relative to the matched single-channel control?",
            "A3_vs_A4": "Does soft boundary encoding help relative to hard masks?",
            "A3_vs_A5": "Does correct geometry matter more than a deliberately shifted mask?",
        },
    }, indent=2))
    print(json.dumps([{key: value for key, value in row.items() if key not in {"history", "label"}} for row in rows], indent=2))


if __name__ == "__main__":
    main()
