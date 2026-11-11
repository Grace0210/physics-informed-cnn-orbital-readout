#!/usr/bin/env python3
"""V3 real-image prediction using six physics-guided input channels."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks


def fft_denoise_fringe(
    image: np.ndarray,
    low_pass_cutoff: float = 0.15,
    notch_threshold: float = 4.0,
) -> np.ndarray:
    """Match cnn_mine/preprocess/preprocess.py's reference-frame FFT filter."""
    shifted = np.fft.fftshift(np.fft.fft2(image))
    spectrum = np.log(np.abs(shifted) + 1.0)
    height, width = image.shape
    center_y, center_x = height // 2, width // 2
    yy, xx = np.ogrid[:height, :width]
    distance = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    high_frequency = distance > int(min(height, width) * low_pass_cutoff)
    threshold = (
        spectrum[high_frequency].mean()
        + notch_threshold * spectrum[high_frequency].std()
    )
    mask = np.ones_like(image, dtype=np.float32)
    mask[(spectrum > threshold) & high_frequency] = 0.0
    return np.abs(
        np.fft.ifft2(np.fft.ifftshift(shifted * mask))
    ).astype(np.float32)


def normalize_by_peak(image: np.ndarray) -> np.ndarray:
    maximum = float(np.max(image))
    if maximum <= 0.0:
        return image.astype(np.float32)
    return (image / maximum).astype(np.float32)


def horizontal_symmetry_axis(smoothed_od: np.ndarray) -> int:
    """Locate the midpoint of a resolved central doublet, or its single peak."""
    height, width = smoothed_od.shape
    profile = smoothed_od[height // 4 : 3 * height // 4].sum(axis=0)
    profile = gaussian_filter1d(profile, sigma=5)

    # The cloud is first located in the central 40% of the camera frame. This
    # excludes bright edge artefacts without assuming the final cropped ROI.
    search_start = int(0.30 * width)
    search_stop = int(0.70 * width)
    local = profile[search_start:search_stop]
    peak_indices, _ = find_peaks(
        local,
        distance=max(12, width // 50),
        prominence=max(float(np.ptp(local)) * 0.04, 1e-8),
    )
    if len(peak_indices) == 0:
        return search_start + int(np.argmax(local))

    peaks = peak_indices + search_start
    primary = int(peaks[np.argmax(profile[peaks])])
    primary_height = float(profile[primary])

    # A double-peaked image must be centered between the two comparable central
    # peaks, not on whichever side happens to be a little brighter.
    partners = [
        int(peak)
        for peak in peaks
        if (
            20 <= abs(int(peak) - primary) <= 140
            and float(profile[peak]) >= 0.65 * primary_height
        )
    ]
    if not partners:
        return primary
    partner = max(partners, key=lambda peak: float(profile[peak]))
    return int(round((primary + partner) / 2.0))


def preprocess_pair(
    signal_path: Path,
    reference_path: Path,
    background_signal: np.ndarray,
    background_reference: np.ndarray,
    crop_height: int = 400,
    crop_width: int = 300,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Use the alignment/cropping/normalization pipeline from cnn_mine."""
    signal = np.asarray(Image.open(signal_path), dtype=np.float32)
    reference = np.asarray(Image.open(reference_path), dtype=np.float32)
    pure_signal = np.clip(signal - background_signal, 1e-5, None)
    pure_reference = np.clip(reference - background_reference, 1e-5, None)

    full_od = np.clip(-np.log(pure_signal / pure_reference), 0.0, None)
    smoothed_od = gaussian_filter(full_od, sigma=5)
    peak_y, _ = np.unravel_index(int(np.argmax(smoothed_od)), smoothed_od.shape)
    symmetry_x = horizontal_symmetry_axis(smoothed_od)

    height, width = signal.shape
    target_y, target_x = height // 2, width // 2
    shift_y = target_y - int(peak_y)
    shift_x = target_x - symmetry_x
    centered_signal = np.roll(
        np.roll(pure_signal, shift_y, axis=0),
        shift_x,
        axis=1,
    )
    centered_reference = np.roll(
        np.roll(pure_reference, shift_y, axis=0),
        shift_x,
        axis=1,
    )

    start_y = target_y - crop_height // 2
    start_x = target_x - crop_width // 2
    crop_signal = centered_signal[
        start_y : start_y + crop_height,
        start_x : start_x + crop_width,
    ]
    crop_reference = centered_reference[
        start_y : start_y + crop_height,
        start_x : start_x + crop_width,
    ]

    raw_od = np.clip(-np.log(crop_signal / crop_reference), 0.0, None)
    filtered_reference = fft_denoise_fringe(crop_reference)
    fft_od = np.clip(
        -np.log(crop_signal / np.clip(filtered_reference, 1e-5, None)),
        0.0,
        None,
    )
    return (
        normalize_by_peak(raw_od),
        normalize_by_peak(fft_od),
        (shift_x, shift_y),
    )


def time_value(path: Path) -> float:
    match = re.match(r"(\d+(?:\.\d+)?)usAveS\.tif$", path.name)
    return float(match.group(1)) if match else float("inf")


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=(
            project_dir
            / "data"
            / "BandMapping_data"
            / "20260511_fix_t4_3us_scan_t1"
        ),
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=project_dir,
    )
    parser.add_argument(
        "--background-dir",
        type=Path,
        help=(
            "Directory containing BG_s.tif and BG_r.tif. "
            "Defaults to SCAN_DIR/BG."
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=(
            project_dir
            / "output_v3"
            / "lattice_cnn_5band_six_channel.pth"
        ),
    )
    parser.add_argument(
        "--mask-delta",
        type=float,
        default=26.0,
        help="Nominal band spacing in pixels used by the soft masks.",
    )
    parser.add_argument(
        "--mask-softness",
        type=float,
        default=4.0,
        help="Logistic boundary softness in pixels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("real_experiment_predictions.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.project_dir))
    from data_generator3 import build_six_channel_input
    from model_v3 import AdvancedLatticeCNN

    background_dir = (
        args.background_dir
        if args.background_dir is not None
        else args.scan_dir / "BG"
    )
    bg_signal = np.asarray(
        Image.open(background_dir / "BG_s.tif"),
        dtype=np.float32,
    )
    bg_reference = np.asarray(
        Image.open(background_dir / "BG_r.tif"),
        dtype=np.float32,
    )
    signal_paths = sorted(args.scan_dir.glob("*usAveS.tif"), key=time_value)
    if not signal_paths:
        raise FileNotFoundError(f"No *usAveS.tif files in {args.scan_dir}")

    raw_images: list[np.ndarray] = []
    fft_images: list[np.ndarray] = []
    shifts: list[tuple[int, int]] = []
    times: list[float] = []
    for signal_path in signal_paths:
        reference_path = signal_path.with_name(
            signal_path.name.replace("AveS.tif", "AveR.tif")
        )
        if not reference_path.exists():
            raise FileNotFoundError(reference_path)
        raw, fft, shift = preprocess_pair(
            signal_path,
            reference_path,
            bg_signal,
            bg_reference,
        )
        raw_images.append(raw)
        fft_images.append(fft)
        shifts.append(shift)
        times.append(time_value(signal_path))

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    model = AdvancedLatticeCNN().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    def predict(images: list[np.ndarray]) -> np.ndarray:
        channels = [
            build_six_channel_input(
                image,
                center_x=image.shape[1] / 2.0,
                delta=args.mask_delta,
                softness=args.mask_softness,
            )
            for image in images
        ]
        batch = torch.from_numpy(np.stack(channels)).to(device)
        with torch.no_grad():
            return model(batch).cpu().numpy()

    raw_predictions = predict(raw_images)
    fft_predictions = predict(fft_images)
    names = ["s", "p", "d", "f", "g"]

    print(f"device={device}")
    print(
        f"v3_channels=full,s,p,d,f,g; "
        f"mask_delta={args.mask_delta:.2f}; "
        f"mask_softness={args.mask_softness:.2f}"
    )
    print("time_us,mode,s,p,d,f,g,sum,shift_x,shift_y")
    for index, time_us in enumerate(times):
        for mode, predictions in [
            ("raw", raw_predictions),
            ("fft", fft_predictions),
        ]:
            values = ",".join(f"{value:.6f}" for value in predictions[index])
            print(
                f"{time_us:g},{mode},{values},"
                f"{predictions[index].sum():.6f},"
                f"{shifts[index][0]},{shifts[index][1]}"
            )
    sensitivity = np.mean(np.abs(raw_predictions - fft_predictions), axis=1)
    print(
        "raw_fft_prediction_mae="
        + ",".join(
            f"{time:g}us:{mae:.6f}"
            for time, mae in zip(times, sensitivity)
        )
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = figure.add_gridspec(3, len(times), height_ratios=[1, 1, 0.9])
    for column, time_us in enumerate(times):
        for row, images, predictions, label in [
            (0, raw_images, raw_predictions, "Raw OD"),
            (1, fft_images, fft_predictions, "FFT reference"),
        ]:
            axis = figure.add_subplot(grid[row, column])
            artist = axis.imshow(
                images[column],
                cmap="jet",
                origin="lower",
                vmin=0.0,
                vmax=0.35,
            )
            prediction_text = " ".join(
                f"{name}:{100 * value:.1f}%"
                for name, value in zip(names, predictions[column])
            )
            axis.set_title(
                f"{time_us:g} us | {label}\n{prediction_text}",
                fontsize=9,
            )
            axis.axis("off")
            figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.02)

    trend_axis = figure.add_subplot(grid[2, :])
    for band_index, name in enumerate(names):
        trend_axis.plot(
            times,
            raw_predictions[:, band_index],
            marker="o",
            linewidth=2,
            label=f"{name} (raw OD)",
        )
    trend_axis.set_xlabel("Time (us)")
    trend_axis.set_ylabel("Predicted orbital fraction")
    trend_axis.set_ylim(0.0, 1.0)
    trend_axis.set_xticks(times)
    trend_axis.grid(alpha=0.3)
    trend_axis.legend(ncol=5)
    trend_axis.set_title(
        "Real experimental predictions; "
        "numbers above images are model outputs, not ground truth"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
