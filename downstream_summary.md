# Downstream PnP/RED stochastic group averaging summary

Last updated: 2026-07-16.

This note summarizes the downstream experiments comparing vanilla classical
denoisers with group-averaged denoisers inside PnP-HQS and RED iterations.

## Experiment setup

- Denoisers: `wavelet`, `tv`.
- Solvers: `pnp_hqs`, `red_gd`.
- Problems: `blur`, `inpaint`.
- Images: first 10 images from `val_images`.
- Noise level: `sigma=15`.
- Iterations: `num_iter=20`.
- Input poses:
  - `upright`: original image, group averaging uses `group_expand=True`.
  - `rot45_padded`: image rotated by 45 degrees and padded to a larger canvas,
    group averaging uses `group_expand=False` to avoid a second canvas expansion.
- Group-averaged modes:
  - `G16_fixed`: full fixed 16-angle averaging.
  - `G4_random`, `G2_random`, `G1_random`: sample 4, 2, or 1 angles per solver
    iteration from the same 16-angle set.
- Baseline:
  - `vanilla`: same solver setup, but the inner denoiser is used without group
    averaging.

Relevant files:

- `scripts/run_downstream_stochastic_wavelet.py`
- `scripts/run_downstream_vanilla_classical.py`
- `results/downstream_stochastic_wavelet/`
- `results/downstream_stochastic_tv/`
- `results/downstream_vanilla_classical/`

All committed CSVs were checked for NaN/inf. The stochastic runs have:

- wavelet: 6400 detail rows, 320 final rows, 32 summary rows.
- TV: 6400 detail rows, 320 final rows, 32 summary rows.
- vanilla baseline: 160 final rows, 16 summary rows.

## Does group averaging improve over vanilla?

### Wavelet

For wavelet, group averaging is usually beneficial, sometimes strongly so.
The main exception is `upright + blur + pnp_hqs`, where full group averaging
hurts.

| input | problem | solver | degraded PSNR | vanilla PSNR | G16 PSNR | G16 - vanilla |
|---|---:|---:|---:|---:|---:|---:|
| rot45_padded | blur | pnp_hqs | 28.89 | 27.87 | 28.49 | +0.62 |
| rot45_padded | blur | red_gd | 28.89 | 30.29 | 31.02 | +0.73 |
| rot45_padded | inpaint | pnp_hqs | 15.46 | 26.80 | 27.70 | +0.90 |
| rot45_padded | inpaint | red_gd | 15.46 | 17.58 | 18.08 | +0.50 |
| upright | blur | pnp_hqs | 23.87 | 23.65 | 22.81 | -0.84 |
| upright | blur | red_gd | 23.87 | 25.74 | 25.76 | +0.02 |
| upright | inpaint | pnp_hqs | 9.73 | 16.60 | 21.89 | +5.28 |
| upright | inpaint | red_gd | 9.73 | 10.06 | 12.35 | +2.29 |

Takeaway: wavelet benefits most for inpainting, especially with upright images.
The `upright + blur + pnp_hqs` setting should not be used as evidence for
improvement; its PnP-HQS parameters appear poorly matched to blur.

### TV

For TV, group averaging gives little to no stable improvement over vanilla.
Differences are usually small, and some upright PnP cases get worse.

| input | problem | solver | degraded PSNR | vanilla PSNR | G16 PSNR | G16 - vanilla |
|---|---:|---:|---:|---:|---:|---:|
| rot45_padded | blur | pnp_hqs | 28.89 | 29.10 | 29.13 | +0.03 |
| rot45_padded | blur | red_gd | 28.89 | 31.51 | 31.52 | +0.01 |
| rot45_padded | inpaint | pnp_hqs | 15.46 | 28.58 | 28.64 | +0.06 |
| rot45_padded | inpaint | red_gd | 15.46 | 19.12 | 19.19 | +0.07 |
| upright | blur | pnp_hqs | 23.87 | 23.88 | 23.66 | -0.22 |
| upright | blur | red_gd | 23.87 | 26.43 | 26.37 | -0.05 |
| upright | inpaint | pnp_hqs | 9.73 | 23.58 | 23.04 | -0.54 |
| upright | inpaint | red_gd | 9.73 | 13.45 | 13.47 | +0.02 |

Takeaway: TV is already close to rotation-stable for this downstream behavior,
or the solver dynamics dominate the denoiser's non-equivariance. Full group
averaging is not clearly worthwhile for TV in this grid.

## Stochastic group averaging vs full G16

### Wavelet

Wavelet is sensitive to the number of sampled angles, but `G4_random` is often
close to `G16_fixed`.

Notable deltas relative to `G16_fixed`:

- `rot45_padded + blur + pnp_hqs`: `G4 -0.13 dB`, `G2 -0.21 dB`, `G1 -0.41 dB`.
- `rot45_padded + inpaint + pnp_hqs`: all stochastic modes within about
  `0.11 dB` of G16.
- `upright + blur + red_gd`: stochastic modes are slightly higher than G16
  (`G2 +0.23 dB`).
- `upright + inpaint + red_gd`: stochastic modes degrade more visibly
  (`G4 -0.25 dB`, `G2 -0.58 dB`, `G1 -0.91 dB`).

Recommendation: for wavelet, `G4_random` is the best cost-accuracy compromise.
`G1_random` is sometimes acceptable, but can lose nearly 1 dB in the harder RED
inpainting setting.

### TV

TV is almost insensitive to stochastic averaging:

- Most stochastic-vs-G16 differences are within `0.02 dB`.
- Even `G1_random` usually matches G16 within measurement noise.

Recommendation: if using TV, stochastic averaging with `G1_random` is enough;
full `G16_fixed` is not justified by PSNR in this experiment.

## Why is final PSNR sometimes not higher than degraded PSNR?

This mainly happens for `blur + pnp_hqs`.

Examples:

- wavelet, upright blur PnP:
  - degraded: `23.87 dB`
  - vanilla final: `23.65 dB`
  - G16 final: `22.81 dB`
- TV, upright blur PnP:
  - degraded: `23.87 dB`
  - vanilla final: `23.88 dB`
  - G16 final: `23.66 dB`

This suggests the current PnP-HQS hyperparameters are not good for blur. The
blurred observation already has relatively high PSNR, and the current denoising
step can over-smooth rather than deblur. This is a solver-parameter issue, not
direct evidence against group averaging.

By contrast, inpainting and RED blur show clear restoration gains:

- wavelet, upright inpaint PnP:
  - degraded: `9.73 dB`
  - vanilla final: `16.60 dB`
  - G16 final: `21.89 dB`
- TV, upright inpaint PnP:
  - degraded: `9.73 dB`
  - vanilla final: `23.58 dB`
  - G16 final: `23.04 dB`
- TV, rot45 blur RED:
  - degraded: `28.89 dB`
  - vanilla final: `31.51 dB`
  - G16 final: `31.52 dB`

## Current interpretation

1. Group averaging helps downstream reconstruction when the denoiser's
   non-equivariance materially affects the solver, as with wavelet inpainting.
2. The effect is denoiser-dependent. TV has little to gain here.
3. Stochastic group averaging is a practical approximation: `G4_random` works
   well for wavelet, and `G1_random` is enough for TV.
4. `blur + pnp_hqs` needs better solver parameters before it can be used as a
   clean demonstration of downstream restoration gains.
5. The `rot45_padded` branch should keep `group_expand=False` during group
   averaging; otherwise the padded image is expanded again and the experiment
   becomes dominated by artificial canvas growth.

