# Orbital-fraction CNN ablation study

This directory is the reproducible experiment package for a controlled ablation:
`A0` white-noise single-channel baseline, `A1` real/PCA noise, `A2` consistency loss,
`A3` six-channel soft masks, `A4` six-channel hard masks, and `A5` deliberately
misregistered soft masks.  A3/A4/A5 have identical six-channel backbones and noise/
consistency settings; A4 tests boundary smoothing and A5 is the negative control.

Run inside the environment that contains PyTorch:

```bash
python run_ablation.py --all
```

For the publication-relevant matched comparison only, train the two six-channel
models below.  A3 and A4 use the same three seeds, full 400x300 images,
architecture, noise, consistency loss, data split, optimizer, and mask-geometry
jitter.  The only changed variable is soft versus hard mask boundaries:

```bash
python run_ablation.py --all --variants A3 A4
```

Laboratory data are not included in this repository. By default the scripts
look for `../data`; set `ORBITAL_DATA_ROOT` to the directory containing
`BandMapping_data/` and the noise banks, and optionally `ORBITAL_PARAMETERS`
to the theory-parameter bank created by `python data_generator2.0.py`.

After the checkpoints exist, evaluate sensitivity to centre and band-spacing
errors without changing the image or its label:

```bash
python evaluate_mask_robustness.py
python evaluate_region_counterfactual.py
```

If only CPU is available, run the bounded feasibility pilot instead. It compares
the relevant A2--A5 variants at half resolution with one seed; it is explicitly
not a substitute for the full study:

```bash
python run_cpu_mask_pilot.py
```

The run trains 18 models (6 variants x 3 seeds), evaluates all checkpoints on one
fixed 750-example benchmark, evaluates the two real scans, and writes English
tables and figures under `results/`. Real-image comparisons are explicitly not
real-data accuracy measurements because no independent orbital ground truth exists.
