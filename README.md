# Orbit Averaging for (Non-)Equivariant Image Denoisers

Reference code for the experiments studying whether **orbit (group) averaging**
makes common image denoisers more equivariant and lowers their MSE.

The estimator under study, for a denoiser `D`, a group `G` acting by `T_g`, and an
input `x`:

```
w(x) = (1/|G|) Σ_g  T_g⁻¹ D(T_g x)          # orbit-averaged estimate
```

## Research questions

> **Q1.** Are the common denoisers used in PnP/RED/diffusion pipelines actually
> non-equivariant, and by how much?
>
> **Q2.** Does orbit averaging consistently improve MSE/PSNR, and is the
> improvement well predicted by the identity
> `SE_avg(x) = E_g SE(T_g x) − e₁(x)`?
>
> **Q3.** Under realistic conditions (rectangular images, discrete rotations,
> fixed-orientation noise) where the strict assumptions of the orbit-MSE
> corollary are violated, does the empirical improvement degrade gracefully?

Each maps to one experiment module / config / runner:

| Question | module | config | what it computes |
|---|---|---|---|
| Q1 | `groupavg.experiments.q1_equivariance` | `configs/q1_equivariance.yaml` | per-element & mean **relative equivariance error** `‖T_g⁻¹D(T_g y) − D(y)‖²/‖D(y)‖²`; `clean_mode` isolates interpolation error |
| Q2 | `groupavg.experiments.q2_orbit_averaging` | `configs/q2_orbit_averaging.yaml` | `EhSE = E_g SE(T_g x)`, `SE_avg`, `e₁`, and the identity check `EhSE − SE_avg ≈ e₁` |
| Q3 | `groupavg.experiments.q3_degradation` | `configs/q3_degradation.yaml` | the Q2 gain swept over violation axes (image shape, exact vs lossy rotation, `\|G\|`, noise support) |

## Layout

```
src/groupavg/
  group_operators.py      # GroupOperator ABC + concrete groups (validated, vendored)
  registry.py             # make_group(name, **kw)
  denoisers/              # Denoiser ABC + registry
    classical.py          #   tv, wavelet, nlm, bm3d(optional)
    restormer.py          #   Restormer (arch vendored, weights referenced)
    stubs.py              #   dncnn, drunet, diffusion  (reserved)
    _restormer_arch.py    #   vendored Restormer architecture
  masks.py                # content / circle / rectangle masks
  metrics.py              # l2sq, psnr, se_to_psnr, orbit_variance (e₁)
  data.py                 # grayscale [0,1] loading + masked Gaussian noise
  pipeline.py             # orbit_average / orbit_estimates / denoise_one
  experiments/            # q1_equivariance, q2_orbit_averaging, q3_degradation
  config.py               # YAML loader (${ref_root} interpolation) + denoiser builder
configs/                  # base + per-question + smoke
scripts/                  # run_experiment.py, make_figures.py
tests/                    # round-trip fidelity, SE_avg identity, denoiser smoke
notebooks/                # result visualization
results/                  # experiment outputs (gitignored)
```

### Group operators (`registry.make_group`)

| name | description | invertibility |
|---|---|---|
| `fourier_rotation` | exact FFT 3-shear rotation, cyclic `C_K` | near-exact (~65 dB) |
| `rotation` | cv2 affine rotation (expand canvas) | lossy on high-freq (~30–45 dB) |
| `shift` | circular translation `Z_n × Z_m` | exact |
| `allpass_fft` | Hermitian unit-magnitude FFT filters | exact, energy-preserving |
| `upsample` / `downsample` | Lanczos / area resampling | lossy |

`fourier_rotation` is the default for Q2 because its near-exact invertibility lets
the `SE_avg = EhSE − e₁` identity be checked without interpolation contamination.

## Assets

Weights and datasets are **referenced** from the existing `equivariant_PGD` repo
(not copied). Edit `ref_root` in `configs/base.yaml` if you move them. Required:

- `Restormer/Denoising/pretrained_models/gaussian_gray_denoising_sigma{15,25,50}.pth`
- `Restormer/Denoising/Datasets/{val_images, val_images_circle, val_images_diag_padded, square_val_images}`

> **Note on DnCNN:** the `model_DnCNN_*.pth` files in the reference repo are
> corrupt (HTML, not weights). DnCNN is therefore a stub for now; re-download
> valid weights and implement `denoisers/stubs.py:DnCNN` to enable it.

## Install & run

```bash
# use the existing torch env (torch 1.12.1, cuda) — Restormer needs torch
conda activate restormer37
pip install -e .

# fast end-to-end sanity check (classical denoiser, 2 images, no GPU needed)
python scripts/run_experiment.py --config configs/smoke.yaml

# headline Q2 (Restormer gaussian-gray σ15, exact Fourier rotation, |G|=16)
python scripts/run_experiment.py --config configs/q2_orbit_averaging.yaml

# Q1 / Q3
python scripts/run_experiment.py --config configs/q1_equivariance.yaml
python scripts/run_experiment.py --config configs/q3_degradation.yaml

# figures from the CSVs
python scripts/make_figures.py --results results --out results/figures

# tests
pytest -q
```

Outputs are written to `results/` as `q{1,2,3}_*.csv` (and figures under
`results/figures/`).

## Conventions

- Images are grayscale `float32` in `[0, 1]`.
- `noise_sigma` is on the **0–255** scale (applied as `sigma/255`); classical
  denoisers convert internally, Restormer weights are tied to a fixed σ.
- Squared error / PSNR are computed on a **content mask** so rotation padding does
  not dominate the metric.
