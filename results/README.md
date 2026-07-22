# Results Directory Map

This directory contains generated experiment outputs. Top-level Q1/Q2/Q3
directories are already organized by experiment. Downstream results are being
standardized under `results/downstream/` for new runs.

## Core Experiments

- `q1_equivariance_grid/`: Q1 non-equivariance curves over rotation angles.
- `q1_cardinal_corrected/`: corrected Q1 checks at 0/90/180/270 degrees.
- `q2_orbit_averaging_grid/`: Q2 orbit-averaging denoising grid.
- `q3_degradation_grid/`: Q3 degradation results.
- `figure5_angle_psnr_grid/`: Figure 5 PSNR vs. rotation-angle grid.
- `orbit_averaging_mse_identity/`: identity-check outputs.
- `rotation_operator_quality/`: rotation-operator diagnostics.

## Downstream Results

Current committed downstream outputs were generated before the nested layout was
standardized, so several completed directories still live at the top level:

- `downstream_stochastic_wavelet/`: stochastic group averaging, wavelet,
  sigma 15.
- `downstream_stochastic_wavelet_sigma25/`: stochastic group averaging,
  wavelet, sigma 25.
- `downstream_stochastic_wavelet_sigma50/`: stochastic group averaging,
  wavelet, sigma 50.
- `downstream_stochastic_tv/`: stochastic group averaging, TV, sigma 15.
- `downstream_stochastic_restormer_sigma15/`: stochastic group averaging,
  Restormer, sigma 15.
- `downstream_stochastic_restormer_aug_sigma15/`: stochastic group averaging,
  retrained/augmented Restormer, sigma 15.
- `downstream_vanilla_classical/`: vanilla wavelet/TV sigma 15 baseline.
- `downstream_vanilla_wavelet_sigma25/`: vanilla wavelet sigma 25 baseline.
- `downstream_vanilla_wavelet_sigma50/`: vanilla wavelet sigma 50 baseline.
- `downstream_vanilla_restormer_sigma15/`: vanilla Restormer sigma 15 baseline.
- `downstream_vanilla_restormer_aug_sigma15/`: vanilla retrained/augmented
  Restormer sigma 15 baseline.

New downstream scripts default to this cleaner layout:

- `downstream/stochastic/<denoiser>_sigma<sigma>/`
- `downstream/vanilla/sigma<sigma>/`
- `downstream/solver_sweep/`

## Active Solver Sweep

The current hyperparameter sweep was launched with explicit save directories,
so it is still writing to:

- `downstream_solver_sweep/`: active vanilla-only classical sweep.
- `downstream_solver_sweep_restormer/`: active vanilla-only Restormer sweep.
- `downstream_solver_sweep_smoke/`: smoke-test output for the sweep script.

Do not move the two active sweep directories while their tmux sessions are
running. After they finish, move or copy their selected final outputs into
`downstream/solver_sweep/`.

## Notes

- `logs/` directories are local run logs and are not usually committed.
- `intermediate_images/` contains visual checkpoints for the first saved image.
- Result CSVs named `*_detail.csv`, `*_final.csv`, and `*_summary.csv` are the
  primary machine-readable artifacts.
