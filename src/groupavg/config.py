"""Tiny YAML config loader with ``${ref_root}`` interpolation.

Configs reference the existing assets in the equivariant_PGD repo by path rather
than copying multi-GB weights/datasets. A single ``ref_root`` key is substituted
into any string value containing ``${ref_root}``.
"""
import os

import yaml


def _interp(obj, mapping):
    if isinstance(obj, str):
        for k, v in mapping.items():
            obj = obj.replace("${%s}" % k, str(v))
        return os.path.expanduser(obj)
    if isinstance(obj, dict):
        return {k: _interp(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interp(v, mapping) for v in obj]
    return obj


def load_config(path, overrides=None):
    """Load a YAML config, optionally merging a flat dict of overrides (top level)."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    ref_root = os.path.expanduser(cfg.get("ref_root", ""))
    cfg = _interp(cfg, {"ref_root": ref_root})
    if overrides:
        cfg.update(overrides)
    return cfg


def dataset_path(cfg, name):
    paths = cfg.get("datasets", {})
    if name not in paths:
        raise KeyError(f"Dataset '{name}' not in config. Known: {sorted(paths)}")
    return paths[name]


def restormer_weights(cfg, sigma=15, color=False):
    d = cfg["restormer_weights_dir"]
    kind = "color" if color else "gray"
    return os.path.join(d, f"gaussian_{kind}_denoising_sigma{int(sigma)}.pth")


def build_denoiser(denoiser_cfg, base_cfg):
    """Construct a Denoiser from an experiment config block + base (for paths).

    denoiser_cfg example: {name: restormer, sigma: 15}  or  {name: wavelet}
    """
    from .denoisers import make_denoiser

    d = dict(denoiser_cfg)
    name = d.pop("name")
    if name == "restormer":
        sigma = d.pop("sigma", 15)
        color = d.pop("color", False)
        weights = d.pop("weights", None) or restormer_weights(base_cfg, sigma=sigma, color=color)
        return make_denoiser("restormer", weights=weights, color=color,
                             device=base_cfg.get("device", "cuda:0"), **d), name
    return make_denoiser(name, **d), name
