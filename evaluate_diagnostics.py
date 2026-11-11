#!/usr/bin/env python3
"""Create regression, error-correlation, Grad-CAM, and curve-fit diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.special import erf
from torch.utils.data import DataLoader

from model import AdvancedLatticeCNN


BANDS = ["s", "p", "d", "f", "g"]
BAND_COLORS = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def validation_predictions(
    model: AdvancedLatticeCNN,
    generator_module,
    parameter_bank: Path,
    data_root: Path,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = generator_module.OnlineTwoViewDataset(
        parameter_bank,
        data_root,
        split="validation",
        seed=seed,
    )
    dataset.set_epoch(0)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for first, second, labels in loader:
            first_prediction = model(first.to(device))
            second_prediction = model(second.to(device))
            predictions.append(
                (0.5 * (first_prediction + second_prediction)).cpu().numpy()
            )
            truths.append(labels.numpy())
    return np.concatenate(truths), np.concatenate(predictions)


def regression_metrics(
    truths: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for index, band in enumerate(BANDS):
        residual = predictions[:, index] - truths[:, index]
        denominator = np.sum(
            (truths[:, index] - np.mean(truths[:, index])) ** 2
        )
        metrics[band] = {
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "r2": float(1.0 - np.sum(residual**2) / denominator),
            "bias": float(np.mean(residual)),
        }
    return metrics


def save_scatter(
    truths: np.ndarray,
    predictions: np.ndarray,
    metrics: dict[str, dict[str, float]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    for index, (axis, band, color) in enumerate(
        zip(axes.flat, BANDS, BAND_COLORS)
    ):
        axis.scatter(
            truths[:, index],
            predictions[:, index],
            s=13,
            alpha=0.32,
            color=color,
            edgecolors="none",
        )
        axis.plot([0, 1], [0, 1], "--", color="black", linewidth=1)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_aspect("equal")
        axis.set_xlabel("True fraction")
        axis.set_ylabel("Predicted fraction")
        values = metrics[band]
        axis.set_title(
            f"{band}-band\n"
            f"MAE={100 * values['mae']:.2f}%, "
            f"RMSE={100 * values['rmse']:.2f}%, "
            f"R²={values['r2']:.3f}"
        )
        axis.grid(alpha=0.25)
    summary_axis = axes.flat[-1]
    summary_axis.axis("off")
    table_rows = [
        [
            band,
            f"{100 * metrics[band]['mae']:.2f}%",
            f"{100 * metrics[band]['rmse']:.2f}%",
            f"{metrics[band]['r2']:.3f}",
            f"{100 * metrics[band]['bias']:+.2f}%",
        ]
        for band in BANDS
    ]
    summary_axis.table(
        cellText=table_rows,
        colLabels=["Band", "MAE", "RMSE", "R²", "Bias"],
        cellLoc="center",
        loc="center",
    )
    summary_axis.set_title("Held-out validation metrics")
    fig.suptitle(
        "Predicted vs true orbital fractions — 750 held-out theory parameters",
        fontsize=15,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_error_correlation(
    truths: np.ndarray,
    predictions: np.ndarray,
    output: Path,
) -> np.ndarray:
    residuals = predictions - truths
    correlation = np.corrcoef(residuals, rowvar=False)
    fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    artist = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(5), BANDS)
    axis.set_yticks(range(5), BANDS)
    axis.set_xlabel("Prediction error")
    axis.set_ylabel("Prediction error")
    axis.set_title(
        "Signed residual correlation\n"
        "residual = predicted fraction − true fraction"
    )
    for row in range(5):
        for column in range(5):
            value = correlation[row, column]
            axis.text(
                column,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.55 else "black",
            )
    fig.colorbar(artist, ax=axis, label="Pearson correlation")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return correlation


def preprocess_real_scan(
    predict_module,
    scan_dir: Path,
    background_dir: Path,
) -> list[dict[str, object]]:
    background_signal = np.asarray(
        Image.open(background_dir / "BG_s.tif"),
        dtype=np.float32,
    )
    background_reference = np.asarray(
        Image.open(background_dir / "BG_r.tif"),
        dtype=np.float32,
    )
    signal_paths = sorted(
        scan_dir.glob("*usAveS.tif"),
        key=predict_module.time_value,
    )
    samples: list[dict[str, object]] = []
    for signal_path in signal_paths:
        reference_path = signal_path.with_name(
            signal_path.name.replace("AveS.tif", "AveR.tif")
        )
        raw, _, shift = predict_module.preprocess_pair(
            signal_path,
            reference_path,
            background_signal,
            background_reference,
        )
        samples.append(
            {
                "time": predict_module.time_value(signal_path),
                "image": raw,
                "shift": shift,
            }
        )
    return samples


def predict_images(
    model: AdvancedLatticeCNN,
    samples: list[dict[str, object]],
    device: torch.device,
) -> np.ndarray:
    batch = np.stack([sample["image"] for sample in samples])[:, None]
    tensor = torch.from_numpy(batch).to(device)
    model.eval()
    with torch.no_grad():
        return model(tensor).cpu().numpy()


def grad_cam(
    model: AdvancedLatticeCNN,
    image: np.ndarray,
    band_index: int,
    device: torch.device,
) -> np.ndarray:
    activation: torch.Tensor | None = None

    def capture(_module, _inputs, output):
        nonlocal activation
        activation = output
        activation.retain_grad()

    target_layer = model.stage4[3]
    handle = target_layer.register_forward_hook(capture)
    tensor = torch.from_numpy(image[None, None]).to(device)
    model.zero_grad(set_to_none=True)
    prediction = model(tensor)
    prediction[0, band_index].backward()
    handle.remove()
    if activation is None or activation.grad is None:
        raise RuntimeError("Grad-CAM activation/gradient was not captured.")
    weights = activation.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = functional.interpolate(
        cam,
        size=image.shape,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    cam -= cam.min()
    maximum = cam.max()
    if float(maximum) > 0:
        cam /= maximum
    return cam.detach().cpu().numpy()


def save_gradcam(
    model: AdvancedLatticeCNN,
    representatives: list[tuple[str, dict[str, object]]],
    device: torch.device,
    output: Path,
) -> None:
    fig, axes = plt.subplots(
        len(representatives),
        6,
        figsize=(18, 8),
        constrained_layout=True,
    )
    if len(representatives) == 1:
        axes = axes[None, :]
    model.eval()
    for row, (label, sample) in enumerate(representatives):
        image = np.asarray(sample["image"], dtype=np.float32)
        with torch.no_grad():
            prediction = model(
                torch.from_numpy(image[None, None]).to(device)
            )[0].cpu().numpy()
        axes[row, 0].imshow(
            image,
            cmap="gray",
            origin="lower",
            vmin=0,
            vmax=0.35,
            aspect="equal",
        )
        axes[row, 0].set_title(f"{label}\npreprocessed OD")
        axes[row, 0].axis("off")
        for band_index, band in enumerate(BANDS):
            cam = grad_cam(model, image, band_index, device)
            axis = axes[row, band_index + 1]
            axis.imshow(
                image,
                cmap="gray",
                origin="lower",
                vmin=0,
                vmax=0.35,
                aspect="equal",
            )
            axis.imshow(
                cam,
                cmap="jet",
                origin="lower",
                alpha=0.55,
                vmin=0,
                vmax=1,
                aspect="equal",
            )
            axis.set_title(
                f"{band}-band Grad-CAM\nprediction={100*prediction[band_index]:.1f}%"
            )
            axis.axis("off")
    fig.suptitle(
        "Grad-CAM at the final residual block — warm regions influence each output",
        fontsize=15,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def five_band_model(
    x: np.ndarray,
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    delta: float,
    center: float,
    sigma: float,
    offset: float,
) -> np.ndarray:
    factor = 2.0 * np.sqrt(2.0) * sigma

    def term(amplitude: float, multiple: int) -> np.ndarray:
        return amplitude / 2.0 * (
            erf((multiple * delta + 2.0 * (x - center)) / factor)
            + erf((multiple * delta - 2.0 * (x - center)) / factor)
        )

    return (
        term(a, 1)
        + term(b, -1)
        + term(b, 2)
        + term(c, -2)
        + term(c, 3)
        + term(d, -3)
        + term(d, 4)
        + term(e, -4)
        + term(e, 5)
        + offset
    )


def curve_fit_profile(image: np.ndarray) -> dict[str, object]:
    x = np.arange(image.shape[1], dtype=np.float64) - image.shape[1] / 2.0
    profile = image.sum(axis=0).astype(np.float64)
    profile = gaussian_filter1d(profile, sigma=2.0)
    profile /= max(float(profile.max()), 1e-8)
    offset_guess = float(np.median(np.r_[profile[:30], profile[-30:]]))
    lower = [0, 0, 0, 0, 0, 10, -30, 0.5, -0.25]
    upper = [3, 3, 3, 3, 3, 55, 30, 18, 0.75]
    best_parameters = None
    best_sse = float("inf")
    for delta_guess in [15, 20, 24, 30, 38, 46]:
        radii = np.abs(x)
        guesses = []
        for low, high in [
            (0, delta_guess / 2),
            (delta_guess / 2, delta_guess),
            (delta_guess, 1.5 * delta_guess),
            (1.5 * delta_guess, 2 * delta_guess),
            (2 * delta_guess, 2.5 * delta_guess),
        ]:
            mask = (radii >= low) & (radii < high)
            value = float(np.mean(profile[mask])) - offset_guess
            guesses.append(float(np.clip(value, 0.01, 1.5)))
        for sigma_guess in [2.0, 5.0, 9.0]:
            initial = [
                *guesses,
                delta_guess,
                0.0,
                sigma_guess,
                float(np.clip(offset_guess, -0.2, 0.7)),
            ]
            try:
                parameters, _ = curve_fit(
                    five_band_model,
                    x,
                    profile,
                    p0=initial,
                    bounds=(lower, upper),
                    maxfev=30000,
                )
            except (RuntimeError, ValueError):
                continue
            fitted = five_band_model(x, *parameters)
            sse = float(np.sum((profile - fitted) ** 2))
            if sse < best_sse:
                best_sse = sse
                best_parameters = parameters
    if best_parameters is None:
        raise RuntimeError("All five-band curve-fit initializations failed.")
    fitted = five_band_model(x, *best_parameters)
    denominator = float(np.sum((profile - profile.mean()) ** 2))
    r2 = 1.0 - best_sse / denominator
    amplitudes = np.clip(best_parameters[:5], 0, None)
    ratios = amplitudes / max(float(amplitudes.sum()), 1e-12)
    return {
        "x": x,
        "profile": profile,
        "fitted": fitted,
        "parameters": best_parameters,
        "ratios": ratios,
        "r2": float(r2),
    }


def save_cnn_curvefit_comparison(
    scan_results: list[dict[str, object]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for column, result in enumerate(scan_results):
        times = np.asarray(result["times"])
        cnn = np.asarray(result["cnn"])
        fit = np.asarray(result["fit"])
        axis = axes[0, column]
        positions = np.arange(len(times))
        width = 0.36
        cnn_bottom = np.zeros(len(times))
        fit_bottom = np.zeros(len(times))
        for band_index, (band, color) in enumerate(zip(BANDS, BAND_COLORS)):
            axis.bar(
                positions - width / 2,
                cnn[:, band_index],
                width,
                bottom=cnn_bottom,
                color=color,
                label=band,
            )
            axis.bar(
                positions + width / 2,
                fit[:, band_index],
                width,
                bottom=fit_bottom,
                color=color,
                alpha=0.55,
                hatch="//",
            )
            cnn_bottom += cnn[:, band_index]
            fit_bottom += fit[:, band_index]
        axis.set_xticks(positions, [f"{time:g} μs" for time in times])
        axis.set_ylim(0, 1)
        axis.set_ylabel("Orbital fraction")
        axis.set_title(
            f"{result['name']}: solid=CNN, hatched=curve fit"
        )
        axis.grid(axis="y", alpha=0.25)
        if column == 0:
            axis.legend(ncol=5, loc="lower center")

        representative_index = int(result["representative_index"])
        fit_result = result["fit_results"][representative_index]
        profile_axis = axes[1, column]
        profile_axis.scatter(
            fit_result["x"],
            fit_result["profile"],
            s=9,
            alpha=0.5,
            color="royalblue",
            label="Integrated OD",
        )
        profile_axis.plot(
            fit_result["x"],
            fit_result["fitted"],
            color="red",
            linewidth=2,
            label="Five-band fit",
        )
        time = times[representative_index]
        profile_axis.set_title(
            f"{result['name']} {time:g} μs traditional fit, "
            f"R²={fit_result['r2']:.3f}"
        )
        profile_axis.set_xlabel("x relative to center (pixels)")
        profile_axis.set_ylabel("Normalized column density")
        profile_axis.grid(alpha=0.25)
        profile_axis.legend()
    fig.suptitle(
        "CNN versus traditional five-band curve fitting on real images",
        fontsize=15,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=project)
    parser.add_argument(
        "--model",
        type=Path,
        default=project / "output_v2/lattice_cnn_5band_online_noise.pth",
    )
    parser.add_argument(
        "--parameter-bank",
        type=Path,
        default=project / "data/training_v2/theory_parameters_5000.npz",
    )
    parser.add_argument("--data-root", type=Path, default=project / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project / "output_v2/diagnostics_delta23_25",
    )
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator_module = load_module(
        "data_generator2_0_diagnostics",
        args.project_dir / "data_generator2.0.py",
    )
    predict_module = load_module(
        "predict_real_diagnostics",
        args.project_dir / "predict_real_experiment.py",
    )
    device = choose_device()
    model = AdvancedLatticeCNN().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"device={device}")

    truths, predictions = validation_predictions(
        model,
        generator_module,
        args.parameter_bank,
        args.data_root,
        device,
        args.seed,
    )
    metrics = regression_metrics(truths, predictions)
    np.savez_compressed(
        args.output_dir / "validation_predictions.npz",
        truths=truths,
        predictions=predictions,
        bands=np.asarray(BANDS),
    )
    save_scatter(
        truths,
        predictions,
        metrics,
        args.output_dir / "predicted_vs_true_scatter.png",
    )
    correlation = save_error_correlation(
        truths,
        predictions,
        args.output_dir / "orbital_error_correlation.png",
    )

    band_root = args.data_root / "BandMapping_data"
    t1_dir = band_root / "20260511_fix_t4_3us_scan_t1"
    t5_dir = band_root / "20260511_fix_t4_3us_scan_t5"
    background_dir = t1_dir / "BG"
    t1_samples = preprocess_real_scan(
        predict_module,
        t1_dir,
        background_dir,
    )
    t5_samples = preprocess_real_scan(
        predict_module,
        t5_dir,
        background_dir,
    )
    t1_cnn = predict_images(model, t1_samples, device)
    t5_cnn = predict_images(model, t5_samples, device)
    save_gradcam(
        model,
        [
            ("scan_t1, 25 μs", t1_samples[-1]),
            ("scan_t5, 4 μs", t5_samples[1]),
        ],
        device,
        args.output_dir / "gradcam_real_images.png",
    )

    scan_results: list[dict[str, object]] = []
    curvefit_json: dict[str, object] = {}
    for name, samples, cnn, representative_index in [
        ("scan_t1", t1_samples, t1_cnn, len(t1_samples) - 1),
        ("scan_t5", t5_samples, t5_cnn, 1),
    ]:
        fit_results = [
            curve_fit_profile(np.asarray(sample["image"])) for sample in samples
        ]
        fit_ratios = np.stack([result["ratios"] for result in fit_results])
        times = np.asarray([sample["time"] for sample in samples])
        scan_results.append(
            {
                "name": name,
                "times": times,
                "cnn": cnn,
                "fit": fit_ratios,
                "fit_results": fit_results,
                "representative_index": representative_index,
            }
        )
        curvefit_json[name] = [
            {
                "time_us": float(time),
                "cnn": cnn[index].tolist(),
                "curve_fit": fit_ratios[index].tolist(),
                "fit_r2": float(fit_results[index]["r2"]),
                "delta_pixels": float(
                    fit_results[index]["parameters"][5]
                ),
                "sigma_pixels": float(
                    fit_results[index]["parameters"][7]
                ),
            }
            for index, time in enumerate(times)
        ]
    save_cnn_curvefit_comparison(
        scan_results,
        args.output_dir / "cnn_vs_traditional_curvefit.png",
    )

    report = {
        "device": str(device),
        "validation_count": int(len(truths)),
        "validation_metrics": metrics,
        "error_correlation": correlation.tolist(),
        "real_image_comparison": curvefit_json,
    }
    (args.output_dir / "diagnostics_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved={args.output_dir}")


if __name__ == "__main__":
    main()
