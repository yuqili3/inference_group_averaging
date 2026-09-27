# SOTA Denoiser Offline Diagnostics

Diagnostics for GS-DRUNet, DRUNet, and DnCNN before downstream SOTA comparison. Full CSV outputs live under `/data2/yuqi/inference_group_averaging/sota_downstream/diagnostics/`; compact summaries are committed here.

## Q1 Equivariance Residual (G=16)

| model | sigma | dataset | mean_rel_err | mean_e1 |
|---|---:|---|---:|---:|
| dncnn | 2 | val_images | 6.856e-05 | 8.658e-06 |
| dncnn | 2 | val_images_circle | 5.975e-05 | 7.715e-06 |
| drunet | 15 | val_images | 2.765e-04 | 2.693e-05 |
| drunet | 15 | val_images_circle | 2.567e-04 | 2.672e-05 |
| gsdrunet | 15 | val_images | 2.748e-04 | 2.723e-05 |
| gsdrunet | 15 | val_images_circle | 2.518e-04 | 2.695e-05 |

## Q2 Identity / Orbit-MSE Diagnostic (G=16)

| model | sigma | dataset | EhSE-SEavg | e | vanilla PSNR | averaged PSNR |
|---|---:|---|---:|---:|---:|---:|
| dncnn | 2 | val_images | 2.829e-05 | 8.771e-06 | 41.23 | 43.28 |
| dncnn | 2 | val_images_circle | 1.209e-05 | 7.725e-06 | 42.17 | 43.14 |
| drunet | 15 | val_images | 4.971e-05 | 3.088e-05 | 31.02 | 31.31 |
| drunet | 15 | val_images_circle | 3.353e-05 | 3.043e-05 | 30.78 | 30.96 |
| gsdrunet | 15 | val_images | 5.152e-05 | 3.124e-05 | 30.97 | 31.26 |
| gsdrunet | 15 | val_images_circle | 3.385e-05 | 3.044e-05 | 30.73 | 30.90 |

## Interpretation

- GS-DRUNet and DRUNet have nearly identical diagnostics: Q1 mean_rel_err is about `2.5e-4` to `2.8e-4`; Q2 `e` is about `3.0e-5` at G=16.
- DnCNN at its native sigma=2 has even smaller Q1 residual, about `6e-5`, and Q2 `e` about `8e-6`.
- These are Restormer/near-equivariant-scale diagnostics, not wavelet-scale diagnostics. The predict-then-verify expectation is therefore that downstream `G16_fixed` and `G1_random` gains for these SOTA denoisers should be small; a selected downstream verification is sufficient unless a fixed-canvas e check contradicts this.
