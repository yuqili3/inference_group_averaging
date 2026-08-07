# Denoising circle sanity check

This sanity check isolates pure denoising from the downstream PnP/RED solvers.

Protocol:

- Dataset: `val_images_circle`, first 10 images.
- Noise: additive Gaussian noise with `sigma=15` on the 0-255 scale.
- Geometry: 100 px zero padding before evaluation; input rotations at 30, 45, and 60 degrees without canvas expansion.
- Metrics: PSNR and SSIM on the circle support mask.
- Denoisers: `restormer` and `restormer-aug`.
- Orbit averaging: Fourier rotation group, `G=16`.

## Main Finding

For both Restormer variants, G16 orbit averaging gives almost no pure denoising
gain on the rotated circle images: typically around `0.03-0.09 dB` and about
`0.001` SSIM. This explains why downstream PnP/RED gains for Restormer are small:
there is little denoising-level improvement for the solver to amplify.

The additional no-expand control shows that the large negative upright result
under the downstream policy is caused by using `group_expand=True` for the
already padded upright image. When `group_expand=False` is used for every pose,
the upright loss shrinks to roughly `-0.1 dB`.

## Downstream Policy

This matches the current downstream solver convention: `group_expand=True` for
upright inputs and `group_expand=False` for rotated inputs.

| model | input_pose | group_expand | mean_noisy_psnr | mean_vanilla_psnr | mean_group_avg_psnr | mean_group_avg_minus_vanilla_psnr | mean_group_avg_minus_vanilla_ssim | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| restormer | rot30_padded | False | 24.6161 | 31.1902 | 31.2394 | 0.0492 | 0.0009 | 10.0000 |
| restormer | rot45_padded | False | 24.6106 | 31.1359 | 31.2109 | 0.0750 | 0.0012 | 10.0000 |
| restormer | rot60_padded | False | 24.6260 | 31.1965 | 31.2377 | 0.0411 | 0.0007 | 10.0000 |
| restormer | upright | True | 24.6016 | 31.4602 | 28.3121 | -3.1481 | -0.1153 | 10.0000 |
| restormer-aug | rot30_padded | False | 24.6161 | 31.1477 | 31.2123 | 0.0646 | 0.0010 | 10.0000 |
| restormer-aug | rot45_padded | False | 24.6106 | 31.0948 | 31.1814 | 0.0866 | 0.0014 | 10.0000 |
| restormer-aug | rot60_padded | False | 24.6260 | 31.1569 | 31.2102 | 0.0533 | 0.0009 | 10.0000 |
| restormer-aug | upright | True | 24.6016 | 31.4207 | 29.5465 | -1.8742 | -0.0739 | 10.0000 |

## No-Expand Control

This uses `group_expand=False` for all input poses.

| model | input_pose | group_expand | mean_noisy_psnr | mean_vanilla_psnr | mean_group_avg_psnr | mean_group_avg_minus_vanilla_psnr | mean_group_avg_minus_vanilla_ssim | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| restormer | rot30_padded | False | 24.6161 | 31.1902 | 31.2394 | 0.0492 | 0.0009 | 10.0000 |
| restormer | rot45_padded | False | 24.6106 | 31.1359 | 31.2109 | 0.0750 | 0.0012 | 10.0000 |
| restormer | rot60_padded | False | 24.6260 | 31.1965 | 31.2377 | 0.0411 | 0.0007 | 10.0000 |
| restormer | upright | False | 24.6016 | 31.4602 | 31.3407 | -0.1195 | -0.0009 | 10.0000 |
| restormer-aug | rot30_padded | False | 24.6161 | 31.1477 | 31.2123 | 0.0646 | 0.0010 | 10.0000 |
| restormer-aug | rot45_padded | False | 24.6106 | 31.0948 | 31.1814 | 0.0866 | 0.0014 | 10.0000 |
| restormer-aug | rot60_padded | False | 24.6260 | 31.1569 | 31.2102 | 0.0533 | 0.0009 | 10.0000 |
| restormer-aug | upright | False | 24.6016 | 31.4207 | 31.3081 | -0.1126 | -0.0008 | 10.0000 |

## Interpretation

1. Restormer is already close to rotation-stable on these padded circle images,
   so orbit averaging changes its denoising output only slightly.
2. Restormer-aug behaves similarly; the augmentation/retraining does not create
   a hidden downstream opportunity for orbit averaging in this setting.
3. RED gains are small partly because the denoising prior itself gains little,
   and partly because RED updates are conservative under the selected stable
   parameters.
4. The upright padded-circle case should use `group_expand=False` for orbit
   averaging if the goal is to remove canvas effects; `group_expand=True` adds an
   avoidable geometry mismatch.
