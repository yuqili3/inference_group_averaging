"""Restormer deep denoiser (gaussian gray/color), loaded from a config-referenced
pretrained checkpoint. The architecture is vendored in ``_restormer_arch.py``;
only the weights are referenced by path.
"""
import numpy as np

from .base import Denoiser

# Default network_g config for the gaussian *gray* denoising checkpoints
# (matches Restormer/Denoising/Options/GaussianGrayDenoising_*.yml).
GRAY_NETWORK_G = dict(
    inp_channels=1,
    out_channels=1,
    dim=48,
    num_blocks=[4, 6, 6, 8],
    num_refinement_blocks=4,
    heads=[1, 2, 4, 8],
    ffn_expansion_factor=2.66,
    bias=False,
    LayerNorm_type="BiasFree",
    dual_pixel_task=False,
)

COLOR_NETWORK_G = dict(GRAY_NETWORK_G, inp_channels=3, out_channels=3)


class Restormer(Denoiser):
    name = "restormer"

    def __init__(self, weights, network_g=None, color=False, device="cuda:0",
                 pad_factor=8):
        import torch
        from ._restormer_arch import Restormer as RestormerArch

        self.torch = torch
        self.device = device
        self.pad_factor = pad_factor
        self.color = color
        cfg = dict(network_g) if network_g else (COLOR_NETWORK_G if color else GRAY_NETWORK_G)
        cfg.pop("type", None)

        net = RestormerArch(**cfg)
        ckpt = torch.load(weights, map_location="cpu")
        state = ckpt.get("params", ckpt) if isinstance(ckpt, dict) else ckpt
        net.load_state_dict(state)
        net.to(device).eval()
        self.net = net

    def __call__(self, img01, sigma=15.0):
        """Denoise a single (H,W) grayscale image; sigma is ignored (the weights
        are tied to a fixed noise level)."""
        torch = self.torch
        F = torch.nn.functional
        img = np.asarray(img01, dtype=np.float32)
        if img.ndim == 2:
            img = img[..., None]
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device).float()

        f = self.pad_factor
        h, w = t.shape[2], t.shape[3]
        H, W = ((h + f) // f) * f, ((w + f) // f) * f
        padh, padw = (H - h if h % f else 0), (W - w if w % f else 0)
        t = F.pad(t, (0, padw, 0, padh), "reflect")

        with torch.no_grad():
            torch.cuda.empty_cache()
            out = self.net(t)
        out = out[:, :, :h, :w]
        out = torch.clamp(out, 0, 1).cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
        return out[..., 0] if out.shape[-1] == 1 else out
