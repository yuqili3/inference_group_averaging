"""Q3 — Under realistic conditions that violate the strict assumptions of the
orbit-MSE corollary, does the empirical improvement degrade gracefully?

We sweep the Q2 estimator across "violation axes" and record, per condition:
  realized_gain  = E_x[EhSE_hx - SEavg_x]   (actual MSE reduction from averaging)
  predicted_gain = E_x[e1]                  (corollary's predicted reduction)
  gain_psnr      = SEavg_psnr - EhSE_psnr    (dB improvement of the averaged est.)
  tracking       = realized_gain / predicted_gain  (≈1 if the identity holds)

Violation axes:
  * image shape / support : dataset folder (square vs rectangular vs circle vs padded)
  * discrete rotation     : group ('rotation'=lossy cv2 / 'fourier_rotation'=exact)
                            and orbit size |G| = averaging
  * fixed-orientation noise: noise_mask (confining noise to a fixed support breaks
                            the group-invariance-of-noise assumption)

Graceful degradation = realized_gain stays positive and tracks predicted_gain as
conditions worsen. Writes one tidy grid CSV; returns the DataFrame.
"""
import os

import pandas as pd

from . import q2_orbit_averaging


def run(
    datasets,
    denoiser,
    denoiser_name="restormer",
    groups=("fourier_rotation",),
    averagings=(4, 8, 16),
    noise_sigma=15.0,
    num_noise=4,
    noise_masks=("circle",),
    se_mask="content",
    upsample=1.0,
    max_images=None,
    seed=0,
    save_dir="results",
    save_csv=True,
    verbose=True,
):
    """``datasets`` is a dict {label: path}; the rest are lists of conditions."""
    grid = []
    for ds_label, ds_path in datasets.items():
        for group_name in groups:
            for K in averagings:
                for nmask in noise_masks:
                    if verbose:
                        print(f"[Q3] dataset={ds_label} group={group_name} |G|={K} noise_mask={nmask}")
                    out = q2_orbit_averaging.run(
                        dataset_dir=ds_path,
                        denoiser=denoiser,
                        denoiser_name=denoiser_name,
                        group_name=group_name,
                        averaging=K,
                        upsample=upsample,
                        noise_sigma=noise_sigma,
                        num_noise=num_noise,
                        noise_mask=nmask,
                        se_mask=se_mask,
                        max_images=max_images,
                        seed=seed,
                        save_dir=save_dir,
                        save_csv=False,
                        verbose=False,
                    )
                    s = out["summary"]
                    realized = s["E_x_EhSE_minus_SEavg"]
                    predicted = s["E_x_e1"]
                    grid.append({
                        "denoiser": denoiser_name,
                        "dataset": ds_label, "group": group_name, "averaging": K,
                        "noise_mask": nmask, "noise_sigma": noise_sigma,
                        "EhSE_psnr": s["E_x_EhSE_hx_psnr"],
                        "SEavg_psnr": s["E_x_SEavg_x_psnr"],
                        "gain_psnr": s["E_x_SEavg_x_psnr"] - s["E_x_EhSE_hx_psnr"],
                        "realized_gain": realized,
                        "predicted_gain": predicted,
                        "tracking": (realized / predicted) if abs(predicted) > 1e-12 else float("nan"),
                    })

    df = pd.DataFrame(grid)
    if save_csv:
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(os.path.join(save_dir, f"q3_degradation_denoiser-{denoiser_name}_sigma-{noise_sigma}.csv"), index=False)
    return df
