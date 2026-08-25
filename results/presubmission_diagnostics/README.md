# Pre-submission Diagnostics

These diagnostics address the reviewer/YB questions about interpolation,
noise coloring, and whether the denoising diagnostic measured in one
geometry transfers to the downstream fixed-canvas geometry.

Heavy per-noise/detail CSVs were written outside the repository under
`/data2/yuqi/inference_group_averaging/presubmission_diagnostics/`.

## Exp 1: exact C4 identity

Exact `np.rot90` removes interpolation and padding. The main diagnostic
is whether `EhSE_minus_SEavg` returns to the diagonal with `e1`.

| denoiser | clip_denoised | mean_e1 | mean_gain | mean_abs_error | n |
| --- | --- | --- | --- | --- | --- |
| restormer | False | 5.26912e-06 | 5.26912e-06 | 4.46929e-11 | 20 |
| restormer | True | 5.26912e-06 | 5.26912e-06 | 4.46929e-11 | 20 |
| restormer-aug | False | 5.21193e-06 | 5.21194e-06 | 3.6703e-11 | 20 |
| restormer-aug | True | 5.21193e-06 | 5.21194e-06 | 3.6703e-11 | 20 |
| wavelet | False | 0.000344575 | 0.000344575 | 6.38615e-11 | 20 |
| wavelet | True | 0.000340085 | 0.000340085 | 6.33988e-11 | 20 |

## Exp 2: orbit constancy across input pose

The pose sweep rotates the clean image first and then adds upright
sensor noise. Departures from flat `SE_G` and `e` curves measure
geometry/noise-statistic mismatch in the deployment-like setting.

| denoiser | pose_method | SEavg_range | e1_range | mean_e1 | mean_SEavg_x | n |
| --- | --- | --- | --- | --- | --- | --- |
| restormer | fft3shear | 4.2237e-05 | 1.39331e-06 | 3.16844e-05 | 0.000789574 | 380 |
| restormer | rot90 | 2.667e-06 | 5.15455e-08 | 3.19846e-05 | 0.000760719 | 40 |
| restormer-aug | fft3shear | 4.33836e-05 | 1.2262e-06 | 3.27243e-05 | 0.000792576 | 380 |
| restormer-aug | rot90 | 2.48148e-06 | 8.18477e-09 | 3.24909e-05 | 0.000763359 | 40 |
| wavelet | fft3shear | 6.02069e-05 | 2.34747e-06 | 0.000420607 | 0.00118723 | 380 |
| wavelet | rot90 | 1.63082e-06 | 1.12362e-06 | 0.000420246 | 0.00114667 | 40 |

## Exp 3: fixed-canvas gain vs e

This compares the pure-denoising gain in the downstream fixed-canvas
geometry to the corresponding same-geometry `e`, and contrasts both
with the Fig. 1 rectangle-protocol `e` when available.

| denoiser | mean_fixed_canvas_gain | mean_fixed_canvas_e1 | mean_identity_gain | mean_fig1_rectangle_e1 | mean_vanilla_psnr | mean_avg_psnr |
| --- | --- | --- | --- | --- | --- | --- |
| restormer | 4.07235e-06 | 3.98871e-05 | 3.91649e-05 | 0.000458518 | 30.027 | 30.0354 |
| restormer-aug | 4.18309e-06 | 4.14464e-05 | 4.06721e-05 | 0.000278048 | 30.0121 | 30.0205 |
| wavelet | 0.000364023 | 0.000415628 | 0.0004203 | 0.00121214 | 27.5516 | 28.5555 |
