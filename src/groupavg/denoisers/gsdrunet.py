"""GS-DRUNet denoiser wrapper using the public GSPnP checkpoint.

The original implementation is a PyTorch Lightning module.  For inference we
only need the underlying DRUNet-like ``UNetRes`` architecture and the
``x - Dg(x)`` gradient-step denoising rule used in the GSPnP restoration code.
"""
from pathlib import Path
import sys

import numpy as np

from .base import Denoiser


DEFAULT_GSPNP_ROOT = Path("/data2/yuqi/inference_group_averaging/external/GSPnP")
DEFAULT_WEIGHTS = Path(
    "/data2/yuqi/inference_group_averaging/sota_downstream/weights/GSDRUNet_gray_state.pth"
)


class GSDRUNet(Denoiser):
    name = "gsdrunet"

    def __init__(
        self,
        weights=DEFAULT_WEIGHTS,
        gspnp_root=DEFAULT_GSPNP_ROOT,
        device="cuda:0",
        color=False,
        act_mode="E",
        drunet_nb=2,
        weight_ds=1.0,
        pad_factor=8,
        clip_output=True,
    ):
        import torch

        if color:
            raise NotImplementedError("The current pipeline expects grayscale images.")

        gs_path = Path(gspnp_root) / "GS_denoising"
        if not gs_path.exists():
            raise FileNotFoundError(f"GSPnP GS_denoising folder not found: {gs_path}")
        if str(gs_path) not in sys.path:
            sys.path.insert(0, str(gs_path))

        from models.network_unet import UNetRes  # noqa: WPS433

        self.torch = torch
        self.device = device
        self.weight_ds = float(weight_ds)
        self.pad_factor = int(pad_factor)
        self.clip_output = bool(clip_output)

        self.net = UNetRes(
            in_nc=2,
            out_nc=1,
            nc=[64, 128, 256, 512],
            nb=int(drunet_nb),
            act_mode=act_mode,
            downsample_mode="strideconv",
            upsample_mode="convtranspose",
        )
        ckpt = torch.load(str(weights), map_location="cpu")
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        prefix = "student_grad.model."
        if any(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        self.net.load_state_dict(state, strict=True)
        self.net.to(device).eval()

    def _pad(self, t):
        torch = self.torch
        h, w = t.shape[-2:]
        f = self.pad_factor
        hp = ((h + f - 1) // f) * f
        wp = ((w + f - 1) // f) * f
        padh = hp - h
        padw = wp - w
        if padh or padw:
            t = torch.nn.functional.pad(t, (0, padw, 0, padh), mode="reflect")
        return t, h, w

    def _student_forward(self, x, sigma01):
        torch = self.torch
        x_pad, h, w = self._pad(x)
        noise_level_map = torch.full(
            (x_pad.size(0), 1, x_pad.size(2), x_pad.size(3)),
            float(sigma01),
            dtype=x_pad.dtype,
            device=x_pad.device,
        )
        out = self.net(torch.cat((x_pad, noise_level_map), dim=1))
        return out[..., :h, :w]

    def __call__(self, img01, sigma=15.0):
        torch = self.torch
        img = np.asarray(img01, dtype=np.float32)
        if img.ndim != 2:
            raise ValueError(f"GSDRUNet expects a 2D grayscale image, got {img.shape}")
        x = torch.from_numpy(img).to(self.device).float()[None, None]
        sigma01 = float(sigma) / 255.0

        with torch.enable_grad():
            x_req = x.detach().clone().requires_grad_(True)
            denoised_student = self._student_forward(x_req, sigma01)
            energy = 0.5 * torch.sum((x_req - denoised_student).reshape((x_req.shape[0], -1)) ** 2)
            grad = torch.autograd.grad(energy, x_req, create_graph=False, only_inputs=True)[0]
            out = x_req - self.weight_ds * grad

        if self.clip_output:
            out = torch.clamp(out, 0.0, 1.0)
        return out.detach().cpu().numpy()[0, 0].astype(np.float32)
