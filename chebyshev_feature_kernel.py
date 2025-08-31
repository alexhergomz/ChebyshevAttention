"""
Feature-map Chebyshev kernel K_n(x,y) with rank-2 map:

For x,y in [-1,1], n >= 1:
  K_n(x,y) = T_n(x) T_n(y) + sqrt(1-x^2) sqrt(1-y^2) U_{n-1}(x) U_{n-1}(y)

Define feature map:
  phi_n(x) = [ T_n(x), sqrt(1-x^2) U_{n-1}(x) ] in R^2
Then K_n(x,y) = <phi_n(x), phi_n(y)> is PSD.

Edge case n=0: K_0(x,y) = 1.

This script provides:
  - cheb_T_seq(x, n): T_0..T_n via recurrence
  - cheb_U_seq(x, m): U_0..U_m via recurrence
  - phi_n(x, n): 2-dim feature map per element of x
  - K_n(x, y, n): scalar/batched kernel evaluation via inner product of phi
  - simple tests: PSD Gram check, n=1 sanity, continuity
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def cheb_T_seq(x: torch.Tensor, n: int) -> torch.Tensor:
    """Compute T_0..T_n at x via recurrence.

    Args:
      x: (...,) real tensor assumed in [-1,1] per element
      n: degree >= 0
    Returns:
      T: (..., n+1) where T[...,k] = T_k(x)
    """
    assert n >= 0
    x = x.unsqueeze(-1)  # (..., 1)
    out = torch.empty(x.shape[:-1] + (n + 1,), device=x.device, dtype=x.dtype)
    out[..., 0] = 1.0
    if n >= 1:
        out[..., 1] = x.squeeze(-1)
        for k in range(2, n + 1):
            out[..., k] = 2.0 * x.squeeze(-1) * out[..., k - 1] - out[..., k - 2]
    return out


def cheb_U_seq(x: torch.Tensor, m: int) -> torch.Tensor:
    """Compute U_0..U_m at x via recurrence.

    U_0=1, U_1=2x, U_{k}=2x U_{k-1} - U_{k-2}

    Args:
      x: (...,) in [-1,1]
      m: max index >= 0
    Returns:
      U: (..., m+1)
    """
    assert m >= 0
    x = x.unsqueeze(-1)
    out = torch.empty(x.shape[:-1] + (m + 1,), device=x.device, dtype=x.dtype)
    out[..., 0] = 1.0
    if m >= 1:
        out[..., 1] = 2.0 * x.squeeze(-1)
        for k in range(2, m + 1):
            out[..., k] = 2.0 * x.squeeze(-1) * out[..., k - 1] - out[..., k - 2]
    return out


def phi_n(x: torch.Tensor, n: int) -> torch.Tensor:
    """Return phi_n(x) of shape (..., 2) for n>=1; for n=0 returns [...,1] with 1s.

    x is assumed elementwise in [-1,1]. If input is not normalized, apply
    torch.clamp to preserve domain (optional upstream L2-normalization of heads is preferred).
    """
    if n == 0:
        return torch.ones(x.shape + (1,), device=x.device, dtype=x.dtype)
    x_clamped = torch.clamp(x, -1.0, 1.0)
    Tn = cheb_T_seq(x_clamped, n)[..., n]
    Unm1 = cheb_U_seq(x_clamped, n - 1)[..., n - 1]
    s = torch.sqrt(torch.clamp(1.0 - x_clamped * x_clamped, min=0.0))
    return torch.stack([Tn, s * Unm1], dim=-1)


def K_n(x: torch.Tensor, y: torch.Tensor, n: int) -> torch.Tensor:
    """Kernel value K_n(x,y) via <phi_n(x), phi_n(y)>.

    Broadcasting supported; returns broadcasted shape of x and y.
    """
    phi_x = phi_n(x, n)
    phi_y = phi_n(y, n)
    return (phi_x * phi_y).sum(dim=-1)


def gram_matrix(xs: torch.Tensor, n: int) -> torch.Tensor:
    """Return Gram matrix G_ij = K_n(x_i, x_j). xs shape (N,)."""
    N = xs.size(0)
    px = phi_n(xs, n)  # (N,2) or (N,1)
    return px @ px.transpose(0, 1)


class ChebKernelMixture(torch.nn.Module):
    """Convex mixture over degrees 0..N with softmax weights.

    K_mix(x,y) = sum_{n=0..N} w_n K_n(x,y),   w = softmax(logits)

    Implemented via feature concatenation with sqrt weights:
      psi(x) = concat_n sqrt(w_n) * phi_n(x)  ⇒  K_mix = <psi(x), psi(y)>.
    """

    def __init__(self, max_n: int, *, init_decay: float = 1.0, device=None, dtype=None) -> None:
        super().__init__()
        assert max_n >= 0
        factory_kwargs = {"device": device, "dtype": dtype}
        self.max_n = max_n
        with torch.no_grad():
            base = torch.tensor([-(n * init_decay) for n in range(max_n + 1)], **{k: v for k, v in factory_kwargs.items() if v is not None})
        self.logits = torch.nn.Parameter(base)

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)

    def psi(self, x: torch.Tensor) -> torch.Tensor:
        ws = self.weights  # (N+1,)
        parts = []
        # n=0 term → 1 dim
        if self.max_n >= 0:
            parts.append(torch.sqrt(ws[0]).reshape(1) * phi_n(x, 0)[..., 0:1])
        for n in range(1, self.max_n + 1):
            parts.append(torch.sqrt(ws[n]).reshape(1) * phi_n(x, n))  # (...,2)
        return torch.cat(parts, dim=-1)

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        px = self.psi(x)
        py = self.psi(y)
        return (px * py).sum(dim=-1)

    def gram(self, xs: torch.Tensor) -> torch.Tensor:
        P = self.psi(xs)
        return P @ P.transpose(0, 1)

@torch.no_grad()
def _self_test() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    xs = torch.linspace(-1, 1, steps=33, device=device, dtype=dtype)
    for n in [0, 1, 2, 3, 6]:
        G = gram_matrix(xs, n)
        # PSD check (allow small numerical noise)
        eigmin = torch.linalg.eigvalsh(G).real.min().item()
        print(f"n={n}  Gram min eig ≈ {eigmin:.3e}")
        assert eigmin > -1e-6
    # n=1 sanity: K_1(x,y) = x y + sqrt(1-x^2) sqrt(1-y^2)
    import itertools
    n = 1
    for xv, yv in itertools.product([-1.0, -0.3, 0.0, 0.7, 1.0], repeat=2):
        x = torch.tensor(xv, device=device, dtype=dtype)
        y = torch.tensor(yv, device=device, dtype=dtype)
        lhs = K_n(x, y, n).item()
        rhs = xv * yv + math.sqrt(max(0.0, 1 - xv * xv)) * math.sqrt(max(0.0, 1 - yv * yv))
        assert abs(lhs - rhs) < 1e-6
    # Mixture sanity: K_mix equals sum w_n K_n
    mix = ChebKernelMixture(max_n=4).to(device)
    with torch.no_grad():
        mix.logits.copy_(torch.tensor([0.0, -0.2, -0.4, -0.6, -0.8], device=device))
    w = mix.weights
    x = torch.linspace(-0.9, 0.9, steps=7, device=device, dtype=dtype)
    y = torch.linspace(-0.8, 0.8, steps=5, device=device, dtype=dtype).unsqueeze(-1)
    kmix = mix.kernel(x, y)
    ksum = 0.0
    for n in range(0, 5):
        ksum = ksum + w[n] * K_n(x, y, n)
    assert torch.allclose(kmix, ksum, atol=1e-6)
    print("Self-test passed.")


if __name__ == "__main__":
    _self_test()


