# Project progress — orbit averaging for denoisers

> Resume file. Read this first each session to pick up where we left off.
> Last updated: 2026-07-14.

## Goal
Build a paper-grade reference repo (`inference_group_averaging`) for **line B** of
the research: orbit/group averaging of image denoisers. Paper targets three
questions:

- **Q1** — Are common PnP/RED/diffusion denoisers non-equivariant, and by how much?
- **Q2** — Does orbit averaging consistently improve MSE/PSNR, and is the gain
  predicted by `SE_avg(x) = E_g SE(T_g x) − e₁(x)`?
- **Q3** — Under realistic violations (rectangular images, discrete rotations,
  fixed-orientation noise) does the improvement degrade gracefully?

Source of validated logic: `~/Documents/equivariant_PGD` (notebooks under
`Restormer/Denoising/`, esp. the FOURIER rotation notebook's
`compute_group_averaging_improvement`, and `group_operators.py`).

## Decisions locked in
- Installable Python package, `src/` layout, plain YAML config (no Hydra).
- Denoisers wired now: **Classical (tv/wavelet/nlm/bm3d-optional) + Restormer**.
  DnCNN/DRUNet/diffusion are **stubs** (user said skip DnCNN for now; its local
  `.pth` are corrupt HTML anyway).
- Weights/datasets **referenced** from `equivariant_PGD` via `configs/base.yaml`
  `ref_root` (not copied).
- Deliverable depth: scaffold **and** reproduce a complete Q2 result on real data.

## Environment
- Conda env: **`restormer37`** (`~/anaconda3/envs/restormer37`), py3.7.16,
  torch 1.12.1 + CUDA, 2× A6000 (**GPU 1 is free** → `device: cuda:1`).
  Has numpy/cv2/skimage/einops/pandas/pywt/pytest. **Missing: bm3d** (optional).
- Package installed editable: `restormer37/bin/pip install -e .` (done).
- Run things with `~/anaconda3/envs/restormer37/bin/python`.

## What's built (all written, package imports cleanly)
```
src/groupavg/
  group_operators.py     vendored from equivariant_PGD (validated)
  registry.py            make_group(name): rotation, fourier_rotation[_v2], shift,
                         upsample, downsample, allpass_fft
  denoisers/{base,classical,restormer,stubs,__init__}.py + _restormer_arch.py
  masks.py  metrics.py  data.py  pipeline.py  config.py
  experiments/{q1_equivariance,q2_orbit_averaging,q3_degradation}.py
configs/{base,q1_equivariance,q2_orbit_averaging,q3_degradation,smoke}.yaml
scripts/{run_experiment,make_figures}.py
tests/{test_group_operators,test_metrics,test_denoisers}.py
README.md
```
Key entry point: `python scripts/run_experiment.py --config configs/<x>.yaml`
(reads `experiment: q1|q2|q3` from the config).

## Verified so far
- `pytest -q` → **13 passed** (group round-trip fidelity, SE_avg identity on the
  exact all-pass group, classical denoiser smoke).
- Smoke Q2 (wavelet, 2 imgs, |G|=4, CPU): runs end-to-end, **+0.40 dB** averaging
  gain, identity tracks (`E[EhSE−SEavg]=3.2e-4` vs `E[e1]=4.8e-4`).
- `val_images_circle`: 10 images, 321×321 grayscale.

## GPU check result (Restormer path confirmed)
1 image of val_images_circle, σ15, |G|=16, num_noise=8 → **2m21s** on cuda:1
(so full 10-image run ≈ **~23 min**). Numbers (1 image):
`EhSE_psnr 21.69 → SEavg_psnr 27.35` = **+5.66 dB** averaging gain.

⚠️ **Identity gap to examine (Q2 research point):** for Restormer
`E[EhSE−SEavg]=4.9e-3` but `E[e1]=1.8e-4` — they do NOT match (for wavelet they
roughly did). Need to check whether our `e1` (centered at `D(x_noisy)`, pooled
over noise) is the corollary's `e1`: EhSE uses per-orbit target `T_g x` while
SEavg uses the clean target `x`, so the bias-variance identity isn't being closed
the same way. This is exactly the Q2 "is the gain predicted by the identity?"
question — decide the canonical `e1`/SE definitions before scaling up.

## Status (2026-07-14)
- **Q1 / Q2 / Q3: complete** — experiments run on real data, results committed
  under `results/`, and figure scripts landed in `scripts/make_*.py`
  (PR #1, branch `add-figure-scripts`: identity, protocols, psnr-vs-angle,
  q1-rho, rho-normalized, rotation-operator-quality).
- **Current frontier: downstream use of the orbit-averaged denoiser inside
  inverse-problem solvers** (see next section).

## Downstream: orbit-averaged denoiser in PnP / RED (current frontier)
Goal: show that replacing a denoiser `D` by its orbit-averaged `D_G` *inside a
solver* both (i) improves reconstruction PSNR and (ii) makes the reconstruction
more equivariant. Scope locked in: **inner-denoiser averaging only**, **minimal
solver loops in-repo (no deepinv dependency)** — deepinv used only as a design
reference (`EquivariantDenoiser` == our `orbit_average`).

- **Built & wired, NOT yet run.** `src/groupavg/experiments/downstream_effect.py`
  implements minimal `pnp_hqs` (HQS), `red_gd` (RED), and `diffusion_style` loops
  with `Blur` (FFT Wiener prox) / `Inpaint` (masked prox) physics.
  `_make_denoise_fn(mode="group_avg")` wraps the denoiser with
  `pipeline.orbit_average` (= inference-time group averaging). Records
  reconstruction PSNR (`base_psnr`) and a downstream **equivariance residual**
  (`downstream_equivariance_psnr`) for vanilla vs group_avg.
- **Runner:** `scripts/run_downstream_effect_grid.py` — grid over models
  {restormer, restormer-aug, wavelet, tv, nlm} x sigma {15,25,50} x
  {pnp_hqs, red_gd, diffusion_style} x {blur, inpaint} x angles 0-180.
  Hard-requires CUDA + Restormer weights (`/data2/yuqi/...`) + `val_images` →
  run on the **GPU box**, not the Mac checkout.
- **Figure:** `scripts/make_downstream_effect_figure.py` — two-panel headline
  (left: reconstruction-PSNR gain, vanilla vs `D_G`; right: downstream
  equivariance PSNR vs rotation angle). Reads
  `results/downstream_effect_grid/downstream_effect_grid_summary.csv`.
  Validated against a schema-correct synthetic CSV; PNG only, dpi=400.

### NEXT (resume here)
1. **Run the downstream grid** on the GPU box (start small):
   `~/anaconda3/envs/restormer37/bin/python scripts/run_downstream_effect_grid.py \`
   `  --device cuda:1 --dataset val_images --models wavelet tv restormer \`
   `  --sigmas 15 --problems blur inpaint --algorithms pnp_hqs red_gd --max-images 10`
2. Render the headline: `python scripts/make_downstream_effect_figure.py`
   (`--denoiser restormer --sigma 15 --out figures/downstream_effect.png`).
3. Sanity-check the two claims hold (PSNR gain > 0; equivariance residual PSNR
   higher for `D_G`), then scale the grid to all sigmas/models.
4. Optional: install `bm3d`; re-download valid DnCNN weights + implement
   `denoisers/stubs.py:DnCNN`; add a results notebook under `notebooks/`.

## deepinv reference: transforms & rotation precision (2026-07-14)
Checked deepinv's `transform` API as a design reference for the downstream work
(we do NOT depend on it; in-repo minimal solvers only).
- `Transform.symmetrize(f, average=True)` == our `orbit_average` (Reynolds
  averaging: `forward -> f -> inverse`, averaged over `n_trans`). `inverse()`
  reuses the sampled params and negates them (rotate by `-theta`), so it is an
  *algorithmic* inverse (apply `T` with `-theta`), not a numerical reconstruction.
- `Rotate` uses torchvision `functional.rotate` (default interpolation NEAREST;
  bilinear optional). **Exact only for 90-degree multiples** (pixel permutations,
  C4/D4). For arbitrary continuous angles it interpolates + zero-pads + crops, so
  `T_g^{-1} T_g x != x` — i.e. the Q3 approximate-action regime.
- Precision tier matches our `rotation_operator_quality` benchmark: deepinv
  `Rotate` (bilinear) ~40 dB round-trip vs our default `fourier_rotation`
  (FFT 3-shear) ~65 dB.
- **Decision:** keep `fourier_rotation` for continuous-angle orbit averaging (so
  the `e1` identity stays clean); deepinv would be exact-equivalent only if the
  group is restricted to C4/D4 (`multiples=90` + `Reflect`).

## Notes / gotchas
- `noise_sigma` is on the **0–255** scale everywhere (applied as sigma/255).
- SE/PSNR use a **content mask** (rotation padding excluded). `fourier_rotation`
  is the default for Q2 (near-exact, ~65 dB round-trip) so the identity is clean;
  `rotation` (cv2) is lossy and used in Q3 as a violation axis.
- CSV schema intentionally matches the original `group_avg_improvement_*` files
  (`EhSE_hx`, `SEavg_x`, `e1`, `EhSE_minus_SEavg`, PSNR cols).
- `_psnr` test helper caps at 99.0 (sentinel) → keep round-trip thresholds < 99.
