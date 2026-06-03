# Group Operation Abstractions + Demo (Rotation & Circular Shift)
# This notebook cell defines abstract group operators for image transformations and
# demonstrates them on a synthetic image. It does not depend on external files.

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import unittest


# -----------------------------
# Core abstract class
# -----------------------------
class GroupOperator(ABC):
    """Abstract group operator over images.
    
    Subclasses must implement:
      - ops: a list-like container describing each group element (e.g., angles or shifts)
      - forward(img): apply *all* group elements to img -> list of transformed images
      - invert(idx, img): apply the inverse of the idx-th group element to img
    """

    @property
    @abstractmethod
    def ops(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, img: np.ndarray) -> list:
        raise NotImplementedError

    @abstractmethod
    def invert(self, idx: int, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __len__(self):
        return len(self.ops)

    def transform_all(self, img: np.ndarray) -> list:
        """Alias for forward(img)."""
        return self.forward(img)


# -----------------------------
# Utility helpers
# -----------------------------
def _to_gray_float(img: np.ndarray) -> np.ndarray:
    """Ensure HxW float32 in [0,1]. Accepts HxW or HxWxC uint8/float."""
    if img.ndim == 3:
        # If it's color, convert to gray
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = img[..., 0]
    img = img.astype(np.float32)
    return img

def _affine_rotate_same_size(img: np.ndarray, angle_deg: float, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT):
    """Rotate without expansion (keeps same HxW), consistent with many restoration pipelines."""
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=interpolation, borderMode=borderMode)
    return rotated, M

def _invert_affine_same_size(img: np.ndarray, M: np.ndarray, out_shape):
    Minv = cv2.invertAffineTransform(M)
    h, w = out_shape[:2]
    return cv2.warpAffine(img, Minv, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

def _to_gray_float01(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    return img

def _get_rot_mat_expand(deg: float, w: int, h: int, expand: bool=True):
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, deg, 1.0)
    if expand:
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        bound_w = int(np.round(h * sin + w * cos))
        bound_h = int(np.round(h * cos + w * sin))
        # 平移量：把旋转后的中心平移到新画布中心
        M[0, 2] += ((h * sin + w * cos) / 2) - center[0]
        M[1, 2] += ((h * cos + w * sin) / 2) - center[1]
    else:
        bound_w, bound_h = w, h
    return M, bound_w, bound_h


# -----------------------------
# Concrete groups
# -----------------------------
@dataclass
class RotationGroup(GroupOperator):
    """Cyclic rotation group C_K with expand=True; invert is exact w.r.t. forward."""
    K: int = 8
    interpolation: int = cv2.INTER_LANCZOS4
    borderMode: int = cv2.BORDER_CONSTANT
    expand:bool = True

    _angles: list = None            # [deg_i]
    _mats: list = None              # [2x3 M_i]   (orig -> expanded)
    _sizes: list = None             # [(Bw_i, Bh_i)]
    _orig_hw: tuple = None          # (H, W)

    def __post_init__(self):
        assert self.K >= 1
        self._angles = [i * (360.0 / self.K) for i in range(self.K)]
        self._mats, self._sizes, self._orig_hw = None, None, None

    @property
    def ops(self):
        return self._angles

    def _ensure_for_shape(self, h: int, w: int):
        self._mats, self._sizes = [], []
        self._orig_hw = (h, w)
        for ang in self._angles:
            M, bw, bh = _get_rot_mat_expand(ang, w, h, self.expand)
            self._mats.append(M)
            self._sizes.append((bw, bh))

    def forward(self, img: np.ndarray):
        """返回每个角度的“扩后”图像，尺寸各不相同，但都完整包含原图内容。"""
        img = _to_gray_float01(img)
        h, w = img.shape[:2]
        if self._mats is None or self._orig_hw != (h, w):
            self._ensure_for_shape(h, w)

        outs = []
        for M, (bw, bh) in zip(self._mats, self._sizes):
            out = cv2.warpAffine(img, M, (bw, bh),
                                 flags=self.interpolation,
                                 borderMode=self.borderMode)
            outs.append(out)
        return outs

    def invert(self, idx: int, img: np.ndarray):
        """把“扩后图像”用 M^{-1} 映射回原始尺寸 (W, H)。"""
        assert 0 <= idx < len(self._angles)
        img = _to_gray_float01(img)
        h0, w0 = self._orig_hw
        M = self._mats[idx]
        Minv = cv2.invertAffineTransform(M)
        restored = cv2.warpAffine(img, Minv, (w0, h0),
                                  flags=self.interpolation,
                                  borderMode=self.borderMode)
        return restored


    
# -----------------------------
# FourierRotationGroup helpers
# -----------------------------

def _wrap_deg(deg):
    # map to (-180, 180]
    x = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if x == -180.0 else x


def _pad(img, pad, mode="constant"):
    # pad=((top,bottom),(left,right))
    return np.pad(img, pad, mode=mode)

def _crop(img, pad):
    (t, b), (l, r) = pad
    return img[t:img.shape[0]-b, l:img.shape[1]-r]

def _shear_x_fft(imgc, a):
    # x' = x + a*y  (about center), implemented as per-row phase modulation
    Ny, Nx = imgc.shape
    y  = (np.arange(Ny) - (Ny - 1) / 2.0).reshape(-1, 1)
    kx = np.fft.fftfreq(Nx).reshape(1, -1)
    phase = np.exp(-2j * np.pi * (a * y) * kx)
    F = np.fft.fft(imgc, axis=1)
    return np.fft.ifft(F * phase, axis=1)

def _shear_y_fft(imgc, b):
    # y' = y + b*x (about center), implemented as per-col phase modulation
    Ny, Nx = imgc.shape
    x  = (np.arange(Nx) - (Nx - 1) / 2.0).reshape(1, -1)
    ky = np.fft.fftfreq(Ny).reshape(-1, 1)
    phase = np.exp(-2j * np.pi * ky * (b * x))
    F = np.fft.fft(imgc, axis=0)
    return np.fft.ifft(F * phase, axis=0)

def decompose_theta(th):
    th = _wrap_deg(float(th))
    # choose m so residual r in (-45, 45]
    m = int(np.floor((th + 45.0) / 90.0))
    r = th - 90.0 * m
    # r in (-45,45]
    # optional tiny numerical clean-up:
    if abs(r) < 1e-12: r = 0.0
    return m, r

def _rot90_exact(img, m):
    """
    Exact rotation by m * 90 degrees using array ops (no interpolation).
    m can be negative or large; internally reduced mod 4.
    Positive m => CCW rotation.
    """
    m = int(m) % 4
    if m == 0:
        return img
    return np.rot90(img, k=m)

def rotate_by_mr(img, m, r, eps_deg=1e-6):
    # exact 90° chunk first
    base = np.rot90(img, k=(m % 4))  # CCW positive

    if abs(abs(r) - 180.0) < eps_deg:
        # residual shouldn't hit this range, but keep safe
        return base[::-1, ::-1].copy()

    if abs(r) < eps_deg:
        return base.copy()

    theta = np.deg2rad(r)
    a = -np.tan(theta / 2.0)
    b =  np.sin(theta)

    imgc = base.astype(np.complex64, copy=False)  # faster
    imgc = _shear_x_fft(imgc, a)
    imgc = _shear_y_fft(imgc, b)
    imgc = _shear_x_fft(imgc, a)
    return imgc.real.astype(base.dtype, copy=False)

def rotate_inverse_by_mr(img, m, r, eps_deg=1e-6):
    # inverse of shear part first (reverse order)
    if abs(r) >= eps_deg:
        theta = np.deg2rad(-r)
        a = -np.tan(theta / 2.0)
        b =  np.sin(theta)

        imgc = img.astype(np.complex64, copy=False)
        imgc = _shear_x_fft(imgc, a)
        imgc = _shear_y_fft(imgc, b)
        imgc = _shear_x_fft(imgc, a)
        img = imgc.real.astype(img.dtype, copy=False)

    # then inverse of rot90
    img = np.rot90(img, k=((-m) % 4))
    return img.copy()

def fourier_rotate_noexpand(img, theta_deg, eps_deg=1e-6):
    """
    Rotate by theta_deg on current canvas using:
      exact rot90 decomposition + Fourier 3-shear on residual in [-45,45].
    This avoids tan blow-up near 180.
    """
    m, r = decompose_theta(theta_deg)
    return rotate_by_mr(img, m, r, eps_deg=eps_deg)


@dataclass
class FourierRotationGroup(GroupOperator):
    """Cyclic rotation group C_K using FFT-based 3-shear rotation.

    expand=True:
      - forward: pad to a shared canvas (worst-case over angles), rotate on canvas
      - invert : rotate back on canvas, then crop back to original HxW

    expand=False:
      - forward: rotate directly on original size (no padding)
      - invert : rotate back directly on original size (no cropping)
    """
    K: int = 8
    pad_mode: str = "constant"
    expand: bool = True
    angles: list = None  # optional explicit angles

    _angles: list = None
    _orig_hw: tuple = None
    _pad: tuple = None
    _mr_cache: list = None
    _noexp_pad: tuple = None   # <- 新增：expand=False 时最小方形 pad
    _noexp_n: int = None       # <- 新增：expand=False 时的方形边长

    def __post_init__(self):
        if self.angles is not None:
            self._angles = [float(a) for a in self.angles]
        else:
            assert self.K >= 1
            self._angles = [i * (360.0 / self.K) for i in range(self.K)]

    @property
    def ops(self):
        return self._angles
    
    def _build_square_pad_min(self, h, w):
        """minimal square pad to n=max(h,w), centered"""
        n = int(max(h, w))
        top = (n - h) // 2
        bottom = n - h - top
        left = (n - w) // 2
        right = n - w - left
        return n, ((top, bottom), (left, right))
    
    def _build_shared_pad(self, h, w, tan_cap=10.0, max_canvas=5000):
        n0 = int(np.ceil(np.sqrt(h*h + w*w)))
        n0 = int(np.ceil(n0/2)*2)

        extra = 0
        for ang in self._angles:
            m, r = decompose_theta(ang)

            theta = np.deg2rad(r)
            a = abs(np.tan(theta/2.0))
            b = abs(np.sin(theta))
            a = min(a, tan_cap)

            extra_x = int(np.ceil(a * (n0/2))) + 4
            extra_y = int(np.ceil(b * (n0/2))) + 4
            extra = max(extra, extra_x, extra_y)

        n = n0 + 2 * extra

        if n > max_canvas:
            raise ValueError(f"FourierRotationGroup: required canvas {n} exceeds max_canvas={max_canvas}. "
                             f"Likely due to angles near 180°. Consider removing 180°, increasing max_canvas, "
                             f"or using special-case handling / fallback interpolation for large angles.")

        pad_y0 = (n - h) // 2
        pad_x0 = (n - w) // 2
        return ((pad_y0, n - h - pad_y0), (pad_x0, n - w - pad_x0))

    def forward(self, img: np.ndarray) -> list:
        img = _to_gray_float01(img)
        h, w = img.shape[:2]

        # --- no expand: operate directly on original size ---
        if not self.expand:
            self._orig_hw = (h, w)
            self._mr_cache = []

            # ✅ 对非正方形：pad 到最小方形 n=max(h,w)
            self._noexp_n, self._noexp_pad = self._build_square_pad_min(h, w)
            base = _pad(img, self._noexp_pad, mode=self.pad_mode)  # shape (n,n)

            outs = []
            for ang in self._angles:
                m, r = decompose_theta(ang)
                self._mr_cache.append((m, r))
                y = rotate_by_mr(base, m, r)
                # ✅ crop 回原始 HxW，确保下游尺寸不变
                outs.append(_crop(y, self._noexp_pad))
            return outs

        # --- expand: shared padded canvas ---
        if self._orig_hw != (h, w) or self._pad is None:
            self._orig_hw = (h, w)
            self._pad = self._build_shared_pad(h, w)

        padded = _pad(img, self._pad, mode=self.pad_mode)
        self._mr_cache = []
        outs = []
        for ang in self._angles:
            m, r = decompose_theta(ang)
            self._mr_cache.append((m, r))
            outs.append(rotate_by_mr(padded, m, r))
        return outs

    def invert(self, idx: int, img: np.ndarray) -> np.ndarray:
        assert 0 <= idx < len(self._angles)
        img = _to_gray_float01(img)

        ang = self._angles[idx]

        # --- no expand: invert on original size ---
        if not self.expand:
            if self._mr_cache is None or self._orig_hw is None or self._noexp_pad is None:
                raise RuntimeError("FourierRotationGroup(invert): call forward(img) once first.")

            m, r = self._mr_cache[idx]

            # ✅ 先把 HxW 的 img pad 回 (n,n)，再做逆旋转，再 crop
            img_sq = _pad(img, self._noexp_pad, mode=self.pad_mode)
            back_sq = rotate_inverse_by_mr(img_sq, m, r)
            back = _crop(back_sq, self._noexp_pad)

            # back 已是原始 HxW
            return back

        # --- expand: invert on canvas then crop back ---
        if self._pad is None or self._orig_hw is None:
            raise RuntimeError("FourierRotationGroup(invert): call forward(img) once to initialize pad/orig shape.")

        m, r = self._mr_cache[idx]
        back_padded = rotate_inverse_by_mr(img, m, r)
        return _crop(back_padded, self._pad)


    
@dataclass
class FourierRotationGroup_v2(GroupOperator):
    """Cyclic rotation group C_K using FFT-based 3-shear rotation.
    
    修改版：在 forward 时保留边缘信息，invert 时恢复而非使用 zero padding。
    """
    K: int = 8
    pad_mode: str = "constant"
    expand: bool = True
    angles: list = None

    _angles: list = None
    _orig_hw: tuple = None
    _pad: tuple = None
    _mr_cache: list = None
    _noexp_pad: tuple = None
    _noexp_n: int = None
    
    # 新增：用于缓存 forward 时的完整旋转后画布
    _full_rot_cache: list = None 

    def __post_init__(self):
        if self.angles is not None:
            self._angles = [float(a) for a in self.angles]
        else:
            assert self.K >= 1
            self._angles = [i * (360.0 / self.K) for i in range(self.K)]

    @property
    def ops(self):
        return self._angles
    
    def _build_square_pad_min(self, h, w):
        n = int(max(h, w))
        top = (n - h) // 2
        bottom = n - h - top
        left = (n - w) // 2
        right = n - w - left
        return n, ((top, bottom), (left, right))
    
    def _build_shared_pad(self, h, w, tan_cap=10.0, max_canvas=5000):
        n0 = int(np.ceil(np.sqrt(h*h + w*w)))
        n0 = int(np.ceil(n0/2)*2)

        extra = 0
        for ang in self._angles:
            m, r = decompose_theta(ang)

            theta = np.deg2rad(r)
            a = abs(np.tan(theta/2.0))
            b = abs(np.sin(theta))
            a = min(a, tan_cap)

            extra_x = int(np.ceil(a * (n0/2))) + 4
            extra_y = int(np.ceil(b * (n0/2))) + 4
            extra = max(extra, extra_x, extra_y)

        n = n0 + 2 * extra

        if n > max_canvas:
            raise ValueError(f"FourierRotationGroup: required canvas {n} exceeds max_canvas={max_canvas}. "
                             f"Likely due to angles near 180°. Consider removing 180°, increasing max_canvas, "
                             f"or using special-case handling / fallback interpolation for large angles.")

        pad_y0 = (n - h) // 2
        pad_x0 = (n - w) // 2
        return ((pad_y0, n - h - pad_y0), (pad_x0, n - w - pad_x0))

    def forward(self, img: np.ndarray) -> list:
        img = _to_gray_float01(img)
        h, w = img.shape[:2]
        self._full_rot_cache = [] # 清空缓存

        # --- no expand: 模式下保存边缘 ---
        if not self.expand:
            self._orig_hw = (h, w)
            self._mr_cache = []
            self._noexp_n, self._noexp_pad = self._build_square_pad_min(h, w)
            base = _pad(img, self._noexp_pad, mode=self.pad_mode)

            outs = []
            for ang in self._angles:
                m, r = decompose_theta(ang)
                self._mr_cache.append((m, r))
                full_rot = rotate_by_mr(base, m, r)
                
                # 保存完整的 (n,n) 旋转结果
                self._full_rot_cache.append(full_rot.copy())
                
                # 返回裁剪后的 HxW
                outs.append(_crop(full_rot, self._noexp_pad))
            return outs

        # --- expand: 模式下逻辑类似 ---
        if self._orig_hw != (h, w) or self._pad is None:
            self._orig_hw = (h, w)
            self._pad = self._build_shared_pad(h, w)

        padded = _pad(img, self._pad, mode=self.pad_mode)
        self._mr_cache = []
        outs = []
        for ang in self._angles:
            m, r = decompose_theta(ang)
            self._mr_cache.append((m, r))
            full_rot = rotate_by_mr(padded, m, r)
            
            # 即使在 expand 模式，我们也存一份，或者根据需求决定是否需要存
            self._full_rot_cache.append(full_rot.copy())
            outs.append(full_rot)
        return outs

    def invert(self, idx: int, img: np.ndarray) -> np.ndarray:
        assert 0 <= idx < len(self._angles)
        img = _to_gray_float01(img)
        
        if self._full_rot_cache is None or len(self._full_rot_cache) <= idx:
            raise RuntimeError("FourierRotationGroup(invert): Must call forward first.")

        m, r = self._mr_cache[idx]

        # --- no expand: 关键修改点 ---
        if not self.expand:
            # 1. 获取 forward 时存下的 (n,n) 画布
            full_canvas = self._full_rot_cache[idx].copy()
            
            # 2. 将处理后的 img (HxW) 贴回画布中心区域
            # 获取 padding 尺寸以便定位
            (t, b), (l, r_pad) = self._noexp_pad
            full_canvas[t : t+img.shape[0], l : l+img.shape[1]] = img
            
            # 3. 对含有“新中心+旧边缘”的画布进行逆旋转
            back_sq = rotate_inverse_by_mr(full_canvas, m, r)
            
            # 4. 裁剪回原始 HxW
            return _crop(back_sq, self._noexp_pad)

        # --- expand: 如果 expand 模式下你也做了后续裁剪，逻辑同理 ---
        # 如果 expand 模式没有被下游裁切，直接逆旋转即可
        back_padded = rotate_inverse_by_mr(img, m, r)
        return _crop(back_padded, self._pad)

@dataclass
class CircularShiftGroup(GroupOperator):
    """Group of circular shifts on a 2D grid: Z_{Nx} x Z_{Ny} (subset if you choose fewer steps).
    
    You can specify counts (nx_steps, ny_steps) to create evenly spaced shifts across width/height,
    or pass explicit shift lists via shift_x and shift_y (in pixels).
    """
    nx_steps: int = 4
    ny_steps: int = 4
    shift_x: list = None  # list of integer pixel shifts along x (cols, width)
    shift_y: list = None  # list of integer pixel shifts along y (rows, height)
    _ops: list = None     # list of (dy, dx) tuples

    def __post_init__(self):
        assert self.nx_steps >= 1 and self.ny_steps >= 1

    @property
    def ops(self):
        # Built lazily because we may need image size to generate evenly spaced shifts.
        if self._ops is None:
            raise RuntimeError("Call forward(img) once to build ops for the given image size, or provide shift_x/shift_y.")
        return self._ops

    def _build_ops_for_shape(self, h, w):
        if self.shift_x is not None and self.shift_y is not None:
            sx = [int(v) for v in self.shift_x]
            sy = [int(v) for v in self.shift_y]
        else:
            # Evenly spaced shifts across the full width/height (wrap-around).
            # Example: nx_steps=4 -> shifts at [0, w/4, w/2, 3w/4] rounded.
            sx = [int(round(i * (w / self.nx_steps))) % w for i in range(self.nx_steps)]
            sy = [int(round(j * (h / self.ny_steps))) % h for j in range(self.ny_steps)]
        self._ops = [(dy, dx) for dy in sy for dx in sx]

    def forward(self, img: np.ndarray) -> list:
        img = _to_gray_float(img)
        h, w = img.shape[:2]
        if self._ops is None:
            self._build_ops_for_shape(h, w)
        out = []
        for (dy, dx) in self._ops:
            out.append(np.roll(img, shift=(dy, dx), axis=(0, 1)))
        return out

    def invert(self, idx: int, img: np.ndarray) -> np.ndarray:
        """Inverse of (dy, dx) is (-dy mod h, -dx mod w)."""
        assert self._ops is not None and 0 <= idx < len(self._ops)
        img = _to_gray_float(img)
        dy, dx = self._ops[idx]
        return np.roll(img, shift=(-dy, -dx), axis=(0, 1))

    

# ---------- Upsample group ----------
@dataclass
class UpsampleGroup(GroupOperator):
    """scales >= 1.0; forward: enlarge; invert: resize back to original size."""
    scales: list               # e.g., [1.25, 1.5, 2.0]
    interpolation: int = cv2.INTER_LANCZOS4
    _orig_hw: tuple = None
    _sizes: list = None

    def __post_init__(self):
        assert all(s >= 1.0 for s in self.scales)

    @property
    def ops(self): return self.scales

    def _ensure(self, h, w):
        self._orig_hw = (h, w)
        self._sizes = [(int(round(w*s)), int(round(h*s))) for s in self.scales]

    def forward(self, img):
        img = _to_gray_float01(img); h, w = img.shape
        if self._sizes is None or self._orig_hw != (h, w):
            self._ensure(h, w)
        return [cv2.resize(img, (W, H), interpolation=self.interpolation) for (W, H) in self._sizes]

    def invert(self, idx, img):
        assert 0 <= idx < len(self.scales)
        img = _to_gray_float01(img)
        h0, w0 = self._orig_hw
        return cv2.resize(img, (w0, h0), interpolation=self.interpolation)
    
    
# ---------- Downsample group ----------
@dataclass
class DownsampleGroup(GroupOperator):
    """0 < scales <= 1.0; forward: shrink (AREA); invert: upsample back (Lanczos)."""
    scales: list               # e.g., [0.9, 0.75, 0.5]
    interp_down: int = cv2.INTER_AREA
    interp_up:   int = cv2.INTER_LANCZOS4
    _orig_hw: tuple = None
    _sizes: list = None

    def __post_init__(self):
        assert all(0.0 < s <= 1.0 for s in self.scales)

    @property
    def ops(self): return self.scales

    def _ensure(self, h, w):
        self._orig_hw = (h, w)
        self._sizes = [(int(round(w*s)), int(round(h*s))) for s in self.scales]

    def forward(self, img):
        img = _to_gray_float01(img); h, w = img.shape
        if self._sizes is None or self._orig_hw != (h, w):
            self._ensure(h, w)
        return [cv2.resize(img, (W, H), interpolation=self.interp_down) for (W, H) in self._sizes]

    def invert(self, idx, img):
        assert 0 <= idx < len(self.scales)
        img = _to_gray_float01(img)
        h0, w0 = self._orig_hw
        return cv2.resize(img, (w0, h0), interpolation=self.interp_up)
    
    

    
# ---------- Hermitian helpers ----------
def _hermitize_phase(phase: np.ndarray) -> np.ndarray:
    """
    将任意相位 map (rad) 转为满足 Hermitian 的相位：
      φ(-u,-v) = -φ(u,v)
    并把自共轭点(DC及Nyquist)相位置 0。
    """
    h, w = phase.shape
    # 强制共轭对称：令相位满足 φ(ip,jp) = -φ(i,j) where (ip,jp)=(-i mod h, -j mod w)
    # 只写一半避免互相覆盖(选择字典序较小的一侧)：
    for i in range(h):
        ip = (-i) % h
        for j in range(w):
            jp = (-j) % w
            if (ip < i) or (ip == i and jp < j):
                phase[ip, jp] = -phase[i, j]

    # 自共轭点：必须为 0
    I = [0] + ([h//2] if h % 2 == 0 else [])
    J = [0] + ([w//2] if w % 2 == 0 else [])
    for ii in I:
        for jj in J:
            phase[ii, jj] = 0.0
    return phase

def _make_allpass_filter_from_phase(phase: np.ndarray) -> np.ndarray:
    phase = _hermitize_phase(phase.copy())
    H = np.exp(1j * phase)
    return H.astype(np.complex64, copy=False)

def _make_random_allpass_hermitian(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    phase = rng.uniform(-np.pi/6, np.pi/6, size=(h, w)).astype(np.float32)
    return _make_allpass_filter_from_phase(phase)

def _check_hermitian_unit(H: np.ndarray, tol=1e-6) -> bool:
    if not np.iscomplexobj(H): return False
    if not np.allclose(np.abs(H), 1.0, atol=tol): return False
    h, w = H.shape
    Hp = np.conj(H[::-1, ::-1])  # H(-u,-v) = conj(H(u,v)) in DFT indexing
    Hp = np.roll(Hp, 1, axis=0); Hp = np.roll(Hp, 1, axis=1)  # 对齐 0 索引
    return np.allclose(H, Hp, atol=tol)

# ---------- Phase generators (optional) ----------
def phase_random(h, w):
    return np.random.uniform(-np.pi/6, np.pi/6, size=(h, w)).astype(np.float32)

def phase_none(h, w):
    return np.zeros((h, w)).astype(np.float32)

def phase_linear_shift(h, w, dx, dy):
    """
    线性相位 φ(u,v)=2π( u*dx/h + v*dy/w )，对应时域循环平移 (dy,dx)。
    注意：这是“等价 circular shift”的 all-pass 元素，便于做 sanity check。
    """
    uu, vv = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    phi = 2*np.pi*(uu*dy/h + vv*dx/w)
    return phi.astype(np.float32)

def phase_radial_rings(h, w, k=8):
    """
    同心环相位：φ(r)=2π * floor(r*k)/k，制造分段常相位环带（还是全通）。
    """
    y = np.arange(h) - h/2; x = np.arange(w) - w/2
    Y, X = np.meshgrid(y, x, indexing='ij')
    r = np.sqrt(X**2 + Y**2) / max(h, w)
    phi = 2*np.pi * (np.floor(r*k)/max(k,1))
    return phi.astype(np.float32)


@dataclass
class AllPassFFTGroup(GroupOperator):
    """
    Group of all-pass (unit magnitude) frequency responses with Hermitian symmetry.
      forward:  y = Re[ IFFT( FFT(x) * H_k ) ]
      inverse:  x̂ = Re[ IFFT( FFT(y) * H_k^* ) ]
    """
    n_filters: int = 8                 # 若未提供 filters_f，则按此数目随机生成
    filters_f: list = None             # 可选：直接传入频域滤波器列表（复数、单位幅度、Hermitian）
    seed: int = 0
    clip01: bool = True
    # 可选：内置相位生成器（若不传 filters_f）
    phase_mode: str = "random"         # "random" | "linear" | "rings" | "custom"
    phase_kwargs: dict = None          # linear: {"dx":int,"dy":int}; rings: {"k":int}; custom: {"phase":np.ndarray}

    _ops_: list = None
    _Hlist: list = None
    _shape_hw: tuple = None

    def __post_init__(self):
        assert self.n_filters >= 1 or (self.filters_f is not None and len(self.filters_f) > 0)
        self._ops_ = None; self._Hlist = None; self._shape_hw = None
        if self.phase_kwargs is None: self.phase_kwargs = {}

    @property
    def ops(self):
        if self._ops_ is None:
            raise RuntimeError("Call forward(img) once to build filters for the given image size, "
                               "or pass 'filters_f' at construction.")
        return self._ops_

    def _ensure_for_shape(self, h: int, w: int):
        self._shape_hw = (h, w)
        self._ops_ = list(range(self.n_filters if self.filters_f is None else len(self.filters_f)))
        self._Hlist = []

        if self.filters_f is not None:
            for H in self.filters_f:
                assert H.shape == (h, w), f"Filter shape {H.shape} != {(h,w)}"
                assert _check_hermitian_unit(H), "Provided filters must be unit magnitude & Hermitian."
                self._Hlist.append(H.astype(np.complex64, copy=False))
            return

        # 需要随机/构造
#         rng = np.random.default_rng(self.seed)
        for k in range(self.n_filters):
            if self.phase_mode == "random":
                phi = phase_random(h, w)
            elif self.phase_mode == "none":
                phi = phase_none(h, w)
            elif self.phase_mode == "linear":
                dx = int(self.phase_kwargs.get("dx", 0))
                dy = int(self.phase_kwargs.get("dy", 0))
                phi = phase_linear_shift(h, w, dx, dy)
            elif self.phase_mode == "rings":
                kval = int(self.phase_kwargs.get("k", 8))
                phi = phase_radial_rings(h, w, k=kval)
            elif self.phase_mode == "custom":
                assert "phase" in self.phase_kwargs, "phase_mode='custom' requires phase in phase_kwargs"
                _phase = self.phase_kwargs["phase"]
                assert _phase.shape == (h, w), "custom phase shape mismatch"
                phi = _phase.astype(np.float32)
            else:
                raise ValueError(f"Unknown phase_mode={self.phase_mode}")
            H = _make_allpass_filter_from_phase(phi)
            self._Hlist.append(H)

    def _apply(self, img: np.ndarray, H: np.ndarray) -> np.ndarray:
        X = np.fft.fft2(img)
        Y = np.fft.ifft2(X * H)
        out = np.real(Y).astype(np.float32)
        return np.clip(out, 0.0, 1.0) if self.clip01 else out

    def forward(self, img: np.ndarray) -> list:
        img = _to_gray_float01(img)
        h, w = img.shape[:2]
        if self._Hlist is None or self._shape_hw != (h, w):
            self._ensure_for_shape(h, w)
        return [self._apply(img, H) for H in self._Hlist]

    def invert(self, idx: int, img: np.ndarray) -> np.ndarray:
        assert self._Hlist is not None and 0 <= idx < len(self._Hlist)
        img = _to_gray_float01(img)
        Hc = np.conj(self._Hlist[idx])
        return self._apply(img, Hc)

if __name__ == '__main__':

    def _make_synth(h=256, w=256):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        grad = (xx/(w-1) + yy/(h-1)) * 0.5
        cy, cx, r = h//2, w//2, min(h, w)//4
        circle = (((yy-cy)**2 + (xx-cx)**2) <= r**2).astype(np.float32)
        rect = np.zeros_like(grad); rect[h//8:3*h//8, w//8:3*w//8] = 1.0
        img = np.clip(grad*0.6 + 0.3*circle + 0.8*rect, 0, 1)
        return img

    def _psnr(a, b, eps=1e-16):
        a = a.astype(np.float32); b = b.astype(np.float32)
        mse = np.mean((a-b)**2)
        return 99. if mse <= eps else 10*np.log10(1.0/mse)

    class TestAllPassFFTGroup(unittest.TestCase):
        def setUp(self):
            self.img = _make_synth(256,256)

        def test_no_phase_forward_inverse_psnr(self):
            g = AllPassFFTGroup(n_filters=1, seed=123, clip01=False, phase_mode="none")
            y = g.forward(self.img)[0]
            xr = g.invert(0, y)
            p = _psnr(self.img, xr)
            print("No phase shift PSNR fwd->inv:", p)
            self.assertGreaterEqual(p, 60.0)
        
        def test_forward_inverse_psnr(self):
            g = AllPassFFTGroup(n_filters=1, seed=5, clip01=False, phase_mode="random")
            y = g.forward(self.img)[0]
            xr = g.invert(0, y)
            p = _psnr(self.img, xr)
            print("No phase shift PSNR fwd->inv:", p)
            self.assertGreaterEqual(p, 60.0)
            
        def test_real_output(self):
            g = AllPassFFTGroup(n_filters=2, seed=7, clip01=False, phase_mode="rings", phase_kwargs={"k":12})
            outs = g.forward(self.img)
            # 检查虚部很小（通过内部实现应该完全为 0，这里还是验一下数值误差）
            for y in outs:
                self.assertTrue(np.isfinite(y).all())
                self.assertLess(np.abs(y.imag if np.iscomplexobj(y) else 0).max() if np.iscomplexobj(y) else 0, 1e-10)

        def test_energy_preserving(self):
            g = AllPassFFTGroup(n_filters=3, seed=0, clip01=False, phase_mode="linear", phase_kwargs={"dx":3,"dy":5})
            outs = g.forward(self.img)
            e0 = np.linalg.norm(self.img.ravel())
            for y in outs:
                e = np.linalg.norm(y.ravel())
                print("energy diff", abs(e - e0)/e0)
                self.assertLess(abs(e - e0)/e0, 1e-3)  # 0.1% 相对误差阈值

    suite = unittest.TestLoader().loadTestsFromTestCase(TestAllPassFFTGroup)
    unittest.TextTestRunner(verbosity=2).run(suite)

    