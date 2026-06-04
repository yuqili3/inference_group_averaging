# Rotation Operator Quality Experiment

This diagnostic measures how much each rotation operator damages an image before
any denoiser is involved. For a fixed signal `x`, angle `theta`, and rotation
operator `T_g`, it computes:

```text
PSNR(T_g^-1 T_g x, x)
```

The forward rotation uses canvas expansion so the full rotated image remains in
view. The inverse maps the expanded result back to the original image size.

## Signals

The experiment uses one natural image and two noise-related variants:

- `astronaut`: grayscale `skimage.data.astronaut`.
- `astronaut_sigma15`: astronaut plus Gaussian noise with sigma `15 / 255`.
- `pure_gaussian_noise`: clipped white Gaussian noise, centered at `0.5`.

## Operators

- `cv2_affine_rotation`: OpenCV affine rotation with expanded canvas and
  Lanczos interpolation.
- `fourier_rotation`: the repository's FFT three-shear `FourierRotationGroup`
  with `expand=True`.
- `scipy_ndimage_rotate`: SciPy `ndimage.rotate` with `reshape=True` for forward
  expansion and cubic interpolation.

Angles are sampled from 0 to 360 degrees in 5 degree increments by default.
PSNR uses a fixed peak value of 1.0 for all signals, including pure noise.

## Run

From the repository root:

```bash
~/anaconda3/envs/restormer37/bin/python scripts/run_rotation_operator_quality.py
```

Optional arguments:

```bash
~/anaconda3/envs/restormer37/bin/python scripts/run_rotation_operator_quality.py \
  --angle-step 5 \
  --noise-sigma 15 \
  --seed 0 \
  --save-dir results/rotation_operator_quality
```

## Outputs

The default output folder is `results/rotation_operator_quality/`.

- `rotation_operator_quality_all.csv`: all `(signal, operator, angle_deg, psnr)`
  rows.
- `rotation_operator_quality_<signal>.csv`: per-signal CSVs.
- `rotation_operator_quality_<signal>_psnr.png`: one matplotlib figure per
  signal, with one curve per rotation operator.

This experiment is intended to separate rotation/interpolation quality from
denoiser quality. Low round-trip PSNR at a given angle means the operator itself
is introducing distortion that can affect later equivariance or orbit-averaging
experiments.
