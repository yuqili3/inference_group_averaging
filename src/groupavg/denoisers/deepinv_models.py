"""Lightweight wrappers for deepinv pretrained grayscale denoisers."""
from pathlib import Path
import sys

import numpy as np

from .base import Denoiser


WEIGHTS_DIR = Path("/data2/yuqi/inference_group_averaging/sota_downstream/weights")
DEFAULT_GSPNP_ROOT = Path("/data2/yuqi/inference_group_averaging/external/GSPnP")
DEFAULT_DRUNET_WEIGHTS = WEIGHTS_DIR / "drunet_deepinv_gray_finetune_26k.pth"
DEFAULT_DNCNN_WEIGHTS = WEIGHTS_DIR / "dncnn_sigma2_gray.pth"


def _load_unet_res():
    gs_path = DEFAULT_GSPNP_ROOT / "GS_denoising"
    if str(gs_path) not in sys.path:
        sys.path.insert(0, str(gs_path))
    from models.network_unet import UNetRes  # noqa: WPS433

    return UNetRes


class DeepInvDRUNet(Denoiser):
    name = "drunet"

    def __init__(
        self,
        weights=DEFAULT_DRUNET_WEIGHTS,
        device="cuda:0",
        pad_factor=8,
        clip_output=True,
    ):
        import torch

        UNetRes = _load_unet_res()
        self.torch = torch
        self.device = device
        self.pad_factor = int(pad_factor)
        self.clip_output = bool(clip_output)
        self.net = UNetRes(
            in_nc=2,
            out_nc=1,
            nc=[64, 128, 256, 512],
            nb=4,
            act_mode="R",
            downsample_mode="strideconv",
            upsample_mode="convtranspose",
        )
        state = torch.load(str(weights), map_location="cpu")
        state = state.get("state_dict", state) if isinstance(state, dict) else state
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

    def __call__(self, img01, sigma=15.0):
        torch = self.torch
        img = np.asarray(img01, dtype=np.float32)
        if img.ndim != 2:
            raise ValueError(f"DRUNet expects a 2D grayscale image, got {img.shape}")
        x = torch.from_numpy(img).to(self.device).float()[None, None]
        x, h, w = self._pad(x)
        sigma01 = float(sigma) / 255.0
        noise = torch.full((x.size(0), 1, x.size(2), x.size(3)), sigma01, device=x.device, dtype=x.dtype)
        with torch.no_grad():
            out = self.net(torch.cat((x, noise), dim=1))[..., :h, :w]
        if self.clip_output:
            out = torch.clamp(out, 0.0, 1.0)
        return out.detach().cpu().numpy()[0, 0].astype(np.float32)


class _DnCNNNet:
    @staticmethod
    def build(depth=20, nf=64):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_conv = nn.Conv2d(1, nf, kernel_size=3, stride=1, padding=1, bias=True)
                self.conv_list = nn.ModuleList(
                    [nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True) for _ in range(depth - 2)]
                )
                self.out_conv = nn.Conv2d(nf, 1, kernel_size=3, stride=1, padding=1, bias=True)
                self.nl_list = nn.ModuleList([nn.ReLU() for _ in range(depth - 1)])

            def forward(self, x):
                y = self.nl_list[0](self.in_conv(x))
                for i, conv in enumerate(self.conv_list):
                    y = self.nl_list[i + 1](conv(y))
                return self.out_conv(y) + x

        return Net()


class DeepInvDnCNN(Denoiser):
    name = "dncnn"

    def __init__(self, weights=DEFAULT_DNCNN_WEIGHTS, device="cuda:0", clip_output=True):
        import torch

        self.torch = torch
        self.device = device
        self.clip_output = bool(clip_output)
        self.net = _DnCNNNet.build(depth=20, nf=64)
        state = torch.load(str(weights), map_location="cpu")
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        self.net.load_state_dict(state, strict=True)
        self.net.to(device).eval()

    def __call__(self, img01, sigma=15.0):
        torch = self.torch
        img = np.asarray(img01, dtype=np.float32)
        if img.ndim != 2:
            raise ValueError(f"DnCNN expects a 2D grayscale image, got {img.shape}")
        x = torch.from_numpy(img).to(self.device).float()[None, None]
        with torch.no_grad():
            out = self.net(x)
        if self.clip_output:
            out = torch.clamp(out, 0.0, 1.0)
        return out.detach().cpu().numpy()[0, 0].astype(np.float32)
