#!/usr/bin/env python3
"""V3 six-channel five-band training with two-view consistency."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn, optim
from torch.utils.data import DataLoader

from model_v3 import AdvancedLatticeCNN


def load_generator_module(project_dir: Path):
    module_path = project_dir / "data_generator3.py"
    spec = importlib.util.spec_from_file_location(
        "data_generator3",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def two_view_loss(
    first_prediction: torch.Tensor,
    second_prediction: torch.Tensor,
    labels: torch.Tensor,
    consistency_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    supervised = 0.5 * (
        functional.mse_loss(first_prediction, labels)
        + functional.mse_loss(second_prediction, labels)
    )
    consistency = functional.mse_loss(
        first_prediction,
        second_prediction,
    )
    total = supervised + consistency_weight * consistency
    return total, supervised, consistency


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parameter-bank",
        type=Path,
        default=(
            project_dir
            / "data"
            / "training_v3"
            / "theory_parameters_5000.npz"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_dir / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "output_v3",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--consistency-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    generator_module = load_generator_module(project_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset = generator_module.OnlineTwoViewDataset(
        args.parameter_bank,
        args.data_root,
        split="train",
        seed=args.seed,
    )
    validation_dataset = generator_module.OnlineTwoViewDataset(
        args.parameter_bank,
        args.data_root,
        split="validation",
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = choose_device()
    model = AdvancedLatticeCNN().to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=5e-3,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (
        "train_log_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
    )
    model_path = args.output_dir / "lattice_cnn_5band_six_channel.pth"
    history_path = args.output_dir / "history_six_channel.json"

    best_validation = float("inf")
    patience_counter = 0
    history: list[dict[str, float | int]] = []
    with log_path.open("w", encoding="utf-8") as log:
        def report(message: str) -> None:
            print(message)
            log.write(message + "\n")
            log.flush()

        report(f"Device: {device}")
        report(
            f"Train/validation: {len(train_dataset)}/{len(validation_dataset)}"
        )
        report("Input: full image + s/p/d/f/g soft-region channels.")
        report(
            f"Consistency weight: {args.consistency_weight}; "
            "validation noise is fixed and source-disjoint."
        )

        for epoch in range(1, args.epochs + 1):
            train_dataset.set_epoch(epoch - 1)
            model.train()
            running = np.zeros(3, dtype=np.float64)
            seen = 0
            for first, second, labels in train_loader:
                first = first.to(device)
                second = second.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                first_prediction = model(first)
                second_prediction = model(second)
                total, supervised, consistency = two_view_loss(
                    first_prediction,
                    second_prediction,
                    labels,
                    args.consistency_weight,
                )
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch = first.size(0)
                running += np.asarray(
                    [
                        total.item(),
                        supervised.item(),
                        consistency.item(),
                    ]
                ) * batch
                seen += batch
            train_total, train_supervised, train_consistency = running / seen

            model.eval()
            validation_sum = 0.0
            validation_seen = 0
            with torch.no_grad():
                for first, second, labels in validation_loader:
                    first = first.to(device)
                    second = second.to(device)
                    labels = labels.to(device)
                    first_prediction = model(first)
                    second_prediction = model(second)
                    total, _, _ = two_view_loss(
                        first_prediction,
                        second_prediction,
                        labels,
                        args.consistency_weight,
                    )
                    validation_sum += total.item() * first.size(0)
                    validation_seen += first.size(0)
            validation_loss = validation_sum / validation_seen
            scheduler.step()

            history.append(
                {
                    "epoch": epoch,
                    "train_total": float(train_total),
                    "train_supervised": float(train_supervised),
                    "train_consistency": float(train_consistency),
                    "validation_total": float(validation_loss),
                }
            )
            report(
                f"Epoch [{epoch:03d}/{args.epochs:03d}] "
                f"Train={train_total:.6f} "
                f"Supervised={train_supervised:.6f} "
                f"Consistency={train_consistency:.6f} "
                f"Val={validation_loss:.6f}"
            )

            if validation_loss < best_validation:
                best_validation = validation_loss
                patience_counter = 0
                torch.save(model.state_dict(), model_path)
                report("  Saved new best model.")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    report(f"Early stopping at epoch {epoch}.")
                    break

    history_path.write_text(
        json.dumps(
            {
                "best_validation_loss": best_validation,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
