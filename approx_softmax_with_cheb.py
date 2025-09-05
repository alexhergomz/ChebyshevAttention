from __future__ import annotations

import argparse
import math
from typing import Tuple

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def _cheb_T_seq(x: torch.Tensor, n: int) -> torch.Tensor:
    x = x.unsqueeze(-1)
    out = torch.empty(x.shape[:-1] + (n + 1,), device=x.device, dtype=x.dtype)
    out[..., 0] = 1.0
    if n >= 1:
        out[..., 1] = x.squeeze(-1)
        for k in range(2, n + 1):
            out[..., k] = 2.0 * x.squeeze(-1) * out[..., k - 1] - out[..., k - 2]
    return out


def _cheb_U_seq(x: torch.Tensor, m: int) -> torch.Tensor:
    x = x.unsqueeze(-1)
    out = torch.empty(x.shape[:-1] + (m + 1,), device=x.device, dtype=x.dtype)
    out[..., 0] = 1.0
    if m >= 1:
        out[..., 1] = 2.0 * x.squeeze(-1)
        for k in range(2, m + 1):
            out[..., k] = 2.0 * x.squeeze(-1) * out[..., k - 1] - out[..., k - 2]
    return out


def _phi_n(x: torch.Tensor, n: int) -> torch.Tensor:
    if n == 0:
        return torch.ones(x.shape + (1,), device=x.device, dtype=x.dtype)
    x = torch.clamp(x, -1.0, 1.0)
    Tn = _cheb_T_seq(x, n)[..., n]
    Unm1 = _cheb_U_seq(x, n - 1)[..., n - 1]
    s = torch.sqrt(torch.clamp(1.0 - x * x, min=0.0))
    return torch.stack([Tn, s * Unm1], dim=-1)


def _psi_total(x_vec: torch.Tensor, max_n: int, weights: torch.Tensor) -> torch.Tensor:
    parts = []
    p0 = torch.sqrt(weights[0]).reshape(1) * _phi_n(x_vec, 0)[..., 0:1]
    parts.append(p0)
    for n in range(1, max_n + 1):
        pn = torch.sqrt(weights[n]).reshape(1) * _phi_n(x_vec, n)
        parts.append(pn)
    # sum over last-but-one dim (feature per original dim)
    return torch.cat(parts, dim=-1).sum(dim=-2)


def _bessel_I_n_series(n: int, x: torch.Tensor, terms: int = 50) -> torch.Tensor:
    """Modified Bessel I_n(x) via power series (integer n >= 0).

    I_n(x) = sum_{k=0..inf} (1 / (k! (n+k)!)) * (x/2)^{2k+n}
    Computed in torch for moderate x, n. Suitable for visualization use.
    """
    assert n >= 0
    x = x.to(torch.float64)  # improve stability
    half = x / 2.0
    result = torch.zeros_like(x, dtype=torch.float64)
    term = (half ** n) / math.factorial(n)
    result = result + term
    # Accumulate k from 1..terms
    for k in range(1, terms + 1):
        term = term * (half * half) / (k * (n + k))
        result = result + term
    return result.to(dtype=torch.float32)


def _weights_bessel(max_n: int, beta: float, device, dtype) -> torch.Tensor:
    # exp(beta * cos theta) = I0(beta) + 2 * sum_{n>=1} I_n(beta) cos(n theta)
    beta_t = torch.tensor(beta, device=device, dtype=dtype)
    coeffs = []
    for n in range(0, max_n + 1):
        In = _bessel_I_n_series(n, beta_t)
        if n == 0:
            coeffs.append(In)
        else:
            coeffs.append(2.0 * In)
    coeffs_t = torch.stack(coeffs)
    w = coeffs_t / coeffs_t.sum()
    return w


def _weights_geometric(max_n: int, r: float, device, dtype) -> torch.Tensor:
    # Clamp r to (0, 0.9999] to avoid pathological flat/expanding sequences
    r = float(max(min(r, 0.9999), 1e-6))
    ns = torch.arange(0, max_n + 1, device=device, dtype=dtype)
    coeffs = (torch.tensor(r, device=device, dtype=dtype) ** ns)
    w = coeffs / coeffs.sum()
    return w


def _weights_uniform(max_n: int, device, dtype) -> torch.Tensor:
    w = torch.ones(max_n + 1, device=device, dtype=dtype)
    return w / w.sum()


def _weights_onehot(max_n: int, idx: int, device, dtype) -> torch.Tensor:
    idx = max(0, min(max_n, idx))
    w = torch.zeros(max_n + 1, device=device, dtype=dtype)
    w[idx] = 1.0
    return w


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser("Approximate softmax attention with Chebyshev mixture kernel")
    p.add_argument("--S", type=int, default=64, help="sequence length")
    p.add_argument("--d", type=int, default=64, help="head dimension")
    p.add_argument("--beta", type=float, default=4.0, help="Bessel beta for softmax-kernel match")
    p.add_argument("--max_n", type=int, default=16, help="max Chebyshev degree")
    p.add_argument("--decay", type=str, default="bessel", choices=["bessel", "geom", "uniform", "onehot"], help="degree weight scheme")
    p.add_argument("--geom_r", type=float, default=0.8, help="geometric ratio r for geom decay")
    p.add_argument("--onehot_n", type=int, default=1, help="degree index for onehot decay")
    p.add_argument("--tau", type=float, default=None, help="softmax temperature; default sqrt(d)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None, help="output image path; default auto-named")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    S, d = args.S, args.d
    tau = args.tau if args.tau is not None else math.sqrt(d)

    # Random normalized Q,K with mild correlation for realism
    q = torch.randn(S, d, device=device, dtype=dtype)
    k = 0.5 * q + 0.5 * torch.randn(S, d, device=device, dtype=dtype)
    q = F.normalize(q, p=2.0, dim=-1)
    k = F.normalize(k, p=2.0, dim=-1)

    # Softmax attention
    scores_soft = (q @ k.t()) / tau
    A_soft = torch.softmax(scores_soft, dim=-1)

    # Chebyshev mixture weights
    if args.decay == "bessel":
        w = _weights_bessel(args.max_n, args.beta, device, dtype)
    elif args.decay == "geom":
        w = _weights_geometric(args.max_n, args.geom_r, device, dtype)
    elif args.decay == "uniform":
        w = _weights_uniform(args.max_n, device, dtype)
    else:
        w = _weights_onehot(args.max_n, args.onehot_n, device, dtype)

    # Compute mixture kernel scores by summing per-dimension inner products (no cross-dim mixing)
    # K_mix(q_i, k_j) = sum_d [ w0*<phi0(q_id),phi0(k_jd)> + sum_{n>=1} w_n <phi_n(q_id),phi_n(k_jd)> ]
    # Note: n=0 adds a constant d across all pairs; this cancels under row-softmax but kept for consistency.
    # Precompute phi for all n
    # Start with n=0 constant term
    phiQ0 = _phi_n(q, 0)  # (S,d,1) all ones
    phiK0 = _phi_n(k, 0)
    scores_cheb = w[0] * torch.einsum('sdx,tdx->st', phiQ0, phiK0)
    for n in range(1, args.max_n + 1):
        phiQn = _phi_n(q, n)  # (S,d,2)
        phiKn = _phi_n(k, n)
        scores_cheb = scores_cheb + w[n] * torch.einsum('sdf,tdf->st', phiQn, phiKn)
    # Convert scores into attention via softmax with same temperature (β absorbed if desired via beta flag)
    A_cheb = torch.softmax((args.beta / tau) * scores_cheb, dim=-1)

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    im0 = axes[0].imshow(A_soft.detach().cpu().numpy(), aspect="auto", cmap="viridis")
    axes[0].set_title("Softmax attention")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(A_cheb.detach().cpu().numpy(), aspect="auto", cmap="viridis")
    axes[1].set_title(f"Cheb-mix (decay={args.decay})")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    diff = (A_soft - A_cheb).abs()
    im2 = axes[2].imshow(diff.detach().cpu().numpy(), aspect="auto", cmap="magma")
    axes[2].set_title(f"|Difference|  L1={diff.sum().item():.3f}")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    out_path = args.out
    if out_path is None:
        out_path = f"approx_softmax_vs_cheb_{args.decay}_N{args.max_n}_beta{args.beta}.png"
    fig.suptitle(f"S={S}, d={d}, tau={tau:.2f}")
    fig.savefig(out_path, dpi=160)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()


