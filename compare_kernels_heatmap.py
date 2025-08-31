from __future__ import annotations

import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Make repo root importable when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from local_runner.chebyshev_feature_kernel import cheb_T_seq, K_n
except Exception:
    from chebyshev_feature_kernel import cheb_T_seq, K_n


def build_weights(max_n: int, device, *, pattern: str = "decay", decay: float = 0.5, onehot: int | None = None, seed: int | None = None) -> torch.Tensor:
    """Construct convex weights over degrees 0..N.

    pattern:
      - 'decay'  : w_n ∝ exp(−decay·n)
      - 'uniform': w_n = 1/(N+1)
      - 'onehot' : w_k = 1 at degree k (requires onehot)
      - 'random' : softmax of random logits (seed optional)
    """
    N = max_n
    if pattern == "uniform":
        w = torch.ones(N + 1, device=device, dtype=torch.float32)
        return w / w.sum()
    if pattern == "onehot":
        assert onehot is not None and 0 <= onehot <= N
        w = torch.zeros(N + 1, device=device, dtype=torch.float32)
        w[onehot] = 1.0
        return w
    if pattern == "random":
        g = torch.Generator(device=device)
        if seed is not None:
            g.manual_seed(seed)
        logits = torch.randn(N + 1, generator=g, device=device, dtype=torch.float32)
        return torch.softmax(logits, dim=-1)
    # default: decay
    logits = torch.tensor([-(n * decay) for n in range(N + 1)], device=device, dtype=torch.float32)
    return torch.softmax(logits, dim=-1)


@torch.no_grad()
def compare(max_n: int = 6, num_points: int = 128, out_png: str = "kernel_compare.png", *, pattern: str = "decay", decay: float = 0.5, onehot: int | None = None, seed: int | None = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Grid over [-1,1]
    xs = torch.linspace(-1.0, 1.0, steps=num_points, device=device, dtype=dtype)
    ys = xs

    # Mixture weights
    w = build_weights(max_n, device, pattern=pattern, decay=decay, onehot=onehot, seed=seed)

    # New kernel (rank-2 Chebyshev feature map), convex mixture across degrees
    K_new = torch.zeros(num_points, num_points, device=device, dtype=dtype)
    for n in range(0, max_n + 1):
        # K_n supports broadcasting when passing (N,1) and (1,N)
        K_new += w[n] * K_n(xs.unsqueeze(1), ys.unsqueeze(0), n)

    # Previous kernel (T-only): convex mixture of T_n(x) T_n(y) without the U-term
    K_prev = torch.zeros_like(K_new)
    # Precompute T sequence up to max_n for both axes
    T_x = cheb_T_seq(xs, max_n)  # (N, max_n+1)
    T_y = cheb_T_seq(ys, max_n)  # (N, max_n+1)
    for n in range(0, max_n + 1):
        tx = T_x[:, n].unsqueeze(1)  # (N,1)
        ty = T_y[:, n].unsqueeze(0)  # (1,N)
        K_prev += w[n] * (tx * ty)

    # Difference
    K_diff = K_new - K_prev

    # Move to cpu for plotting
    K_new_c = K_new.cpu().numpy()
    K_prev_c = K_prev.cpu().numpy()
    K_diff_c = K_diff.cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    im0 = axs[0].imshow(K_prev_c, origin='lower', extent=[-1,1,-1,1], cmap='viridis')
    axs[0].set_title(f"Prev T-only mix (N={max_n})")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(K_new_c, origin='lower', extent=[-1,1,-1,1], cmap='viridis')
    axs[1].set_title(f"New rank-2 mix (N={max_n})")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(K_diff_c, origin='lower', extent=[-1,1,-1,1], cmap='coolwarm')
    axs[2].set_title("Difference (new - prev)")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xlabel("y")
    axs[0].set_ylabel("x")

    fig.suptitle(f"Chebyshev kernels: prev vs new | pattern={pattern}")
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")

    # Simple numeric summary
    max_abs = float(torch.max(torch.abs(K_diff)).cpu())
    mean_abs = float(torch.mean(torch.abs(K_diff)).cpu())
    print(f"|diff| max = {max_abs:.3e}, mean = {mean_abs:.3e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_n", type=int, default=6)
    ap.add_argument("--pattern", type=str, default="decay", choices=["decay","uniform","onehot","random"])
    ap.add_argument("--decay", type=float, default=0.5)
    ap.add_argument("--onehot", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num_points", type=int, default=128)
    ap.add_argument("--out", type=str, default="kernel_compare.png")
    args = ap.parse_args()
    compare(args.max_n, args.num_points, args.out, pattern=args.pattern, decay=args.decay, onehot=args.onehot, seed=args.seed)


