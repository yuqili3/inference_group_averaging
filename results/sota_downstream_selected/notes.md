# SOTA Selected Downstream Verification

Selected downstream verification for SOTA denoisers after the offline diagnostics predicted small group-averaging gains. Full outputs live under `/data2/yuqi/inference_group_averaging/sota_downstream/downstream_selected/`. This run uses one BSD rectangle image, input angles `0/45/90`, blur and inpainting, PnP-HQS and RED-GD, and modes `vanilla`, `G1_random`, `G16_fixed`.

## Mean Over The Three Poses

| denoiser | problem | algorithm | mode | PSNR | dPSNR vs vanilla | SSIM | dSSIM vs vanilla |
|---|---|---|---|---:|---:|---:|---:|
| dncnn | blur | pnp_hqs | G16_fixed | 23.228 | 0.025 | 0.6616 | 0.0007 |
| dncnn | blur | pnp_hqs | G1_random | 23.226 | 0.023 | 0.6617 | 0.0008 |
| dncnn | blur | pnp_hqs | vanilla | 23.203 | 0.000 | 0.6609 | 0.0000 |
| dncnn | blur | red_gd | G16_fixed | 23.541 | 0.002 | 0.7010 | 0.0000 |
| dncnn | blur | red_gd | G1_random | 23.540 | 0.002 | 0.7010 | 0.0000 |
| dncnn | blur | red_gd | vanilla | 23.539 | 0.000 | 0.7010 | 0.0000 |
| dncnn | inpaint | pnp_hqs | G16_fixed | 12.420 | 2.326 | 0.3145 | 0.1031 |
| dncnn | inpaint | pnp_hqs | G1_random | 12.449 | 2.355 | 0.3152 | 0.1039 |
| dncnn | inpaint | pnp_hqs | vanilla | 10.094 | 0.000 | 0.2114 | 0.0000 |
| dncnn | inpaint | red_gd | G16_fixed | 10.864 | 0.852 | 0.2389 | 0.0393 |
| dncnn | inpaint | red_gd | G1_random | 10.794 | 0.782 | 0.2354 | 0.0357 |
| dncnn | inpaint | red_gd | vanilla | 10.011 | 0.000 | 0.1996 | 0.0000 |
| drunet | blur | pnp_hqs | G16_fixed | 22.085 | -0.005 | 0.5343 | 0.0002 |
| drunet | blur | pnp_hqs | G1_random | 22.088 | -0.002 | 0.5342 | 0.0002 |
| drunet | blur | pnp_hqs | vanilla | 22.090 | 0.000 | 0.5340 | 0.0000 |
| drunet | blur | red_gd | G16_fixed | 23.391 | 0.004 | 0.6821 | 0.0001 |
| drunet | blur | red_gd | G1_random | 23.390 | 0.003 | 0.6821 | 0.0001 |
| drunet | blur | red_gd | vanilla | 23.387 | 0.000 | 0.6821 | 0.0000 |
| drunet | inpaint | pnp_hqs | G16_fixed | 13.862 | 2.796 | 0.4377 | 0.0965 |
| drunet | inpaint | pnp_hqs | G1_random | 13.837 | 2.771 | 0.4332 | 0.0920 |
| drunet | inpaint | pnp_hqs | vanilla | 11.066 | 0.000 | 0.3412 | 0.0000 |
| drunet | inpaint | red_gd | G16_fixed | 10.874 | 0.600 | 0.2582 | 0.0241 |
| drunet | inpaint | red_gd | G1_random | 10.812 | 0.538 | 0.2559 | 0.0218 |
| drunet | inpaint | red_gd | vanilla | 10.274 | 0.000 | 0.2340 | 0.0000 |
| gsdrunet | blur | pnp_hqs | G16_fixed | 22.189 | -0.001 | 0.5377 | -0.0008 |
| gsdrunet | blur | pnp_hqs | G1_random | 22.192 | 0.002 | 0.5378 | -0.0007 |
| gsdrunet | blur | pnp_hqs | vanilla | 22.190 | 0.000 | 0.5385 | 0.0000 |
| gsdrunet | blur | red_gd | G16_fixed | 23.422 | 0.003 | 0.6829 | 0.0000 |
| gsdrunet | blur | red_gd | G1_random | 23.423 | 0.004 | 0.6829 | 0.0000 |
| gsdrunet | blur | red_gd | vanilla | 23.419 | 0.000 | 0.6828 | 0.0000 |
| gsdrunet | inpaint | pnp_hqs | G16_fixed | 14.162 | 2.533 | 0.4508 | 0.0686 |
| gsdrunet | inpaint | pnp_hqs | G1_random | 14.231 | 2.603 | 0.4494 | 0.0672 |
| gsdrunet | inpaint | pnp_hqs | vanilla | 11.629 | 0.000 | 0.3822 | 0.0000 |
| gsdrunet | inpaint | red_gd | G16_fixed | 11.035 | 0.663 | 0.2734 | 0.0193 |
| gsdrunet | inpaint | red_gd | G1_random | 10.977 | 0.606 | 0.2713 | 0.0172 |
| gsdrunet | inpaint | red_gd | vanilla | 10.371 | 0.000 | 0.2541 | 0.0000 |

## Takeaways

- Blur: all three SOTA denoisers show tiny averaging effects in this selected run, usually within about `0.05` dB. This matches the small Q1/Q2 offline diagnostics.
- Inpainting: all three SOTA denoisers show large selected-run gains under PnP-HQS (`+2.3` to `+2.8` dB) and moderate gains under RED-GD (`+0.6` to `+0.85` dB). This does not match a pure denoising-`e` explanation and should be treated as a solver/null-space effect to verify with the larger `0:5:90` sweep.
- `G1_random` often tracks `G16_fixed` surprisingly closely in this selected run; this is useful for the EPnP/ERED comparison, but needs the larger `0:5:90` sweep before making a strong statement.
