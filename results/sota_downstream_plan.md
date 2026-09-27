# SOTA Downstream Comparison Plan

Goal: compare the downstream blur/inpainting solver results against recent
PnP/SOTA denoisers while using the paper's offline diagnostic `e` as a
predict-then-verify check.

## Target Models

- `GSDRUNet`: preferred source is `deepinv.models.GSDRUNet`, whose documented
  pretrained weights come from Hurault et al.'s Gradient-Step PnP/GSPnP work.
  Deepinv lists both color and grayscale weights trained on noise levels
  `[0, 50]/255`.
- `DRUNet`: preferred source is `deepinv.models.DRUNet`; use grayscale weights
  for compatibility with the current grayscale pipeline.
- `DnCNN`: preferred source is `deepinv.models.DnCNN`; note that the default
  documented weights are trained for noise level `2/255`, so Q1/Q2 should be
  measured at sigma=2 and downstream should use the paper's measurement noise
  setting when comparing to the cited method.

## Current Environment Status

- Current experiment environment: `/home/yuqi/anaconda3/envs/restormer37`.
- `deepinv` is not installed in this environment.
- `pip index versions deepinv` on the configured index only exposes
  `deepinv==0.0.1`, while current documentation is for `deepinv 0.4.2`.
- Local `/data2/yuqi` scan did not find an existing GSPnP/DRUNet checkout or
  model weights.
- Some old DnCNN-looking files under `~/Documents/equivariant_PGD` appear to be
  downloaded GitHub HTML pages, not reliable PyTorch checkpoints.

## Implementation Tasks

1. Install or vendor deepinv-compatible denoisers.
   - Preferred: create a modern Python environment on `/data2` or install
     deepinv from GitHub/PyPI if dependency versions allow it.
   - Fallback: clone GSPnP and load its grayscale GS-DRUNet checkpoint directly.
2. Add `deepinv_drunet`, `deepinv_dncnn`, and `deepinv_gsdrunet` wrappers under
   `src/groupavg/denoisers/`.
3. Keep grayscale as the first protocol for fairness with existing BSD68 and
   downstream experiments. Only add color CBSD68 if grayscale GS-DRUNet is not
   usable or the paper comparison requires color.
4. Run offline diagnostics before downstream:
   - Q1 rho sweep: C16 FFT 3-shear plus exact cardinal reference.
   - Q2 identity/e: disk and rectangle protocols.
   - Fixed-canvas `e`: same geometry as downstream.
5. Decision rule:
   - If fixed-canvas `e` is wavelet-scale, run full downstream selected sweep.
   - If fixed-canvas `e` is Restormer-scale, report the predicted near-zero
     downstream gain and verify on a small selected downstream sweep.
6. Downstream SOTA comparison:
   - Use rectangle angle sweep with input angles `0:5:90`.
   - Compare modes `vanilla`, `G1_random`, `G16_fixed`.
   - Problems: blur and inpainting.
   - Solvers: PnP-HQS and RED-GD.

## Code Status

- `scripts/run_downstream_solver_sweep.py` now supports `G1_random`.
- `G1_random` samples one angle from the C16 base grid on every denoiser call,
  then applies `T_g -> D -> T_g^{-1}`. This matches the EPnP/ERED single-random
  group-element estimator at the denoiser-call level.

## Storage Policy

- Full detail CSVs, downloaded weights, cloned external repos, and intermediate
  images should go under `/data2/yuqi/inference_group_averaging/sota_downstream/`.
- Repo commits should contain only scripts, compact summaries, plots, and notes.
